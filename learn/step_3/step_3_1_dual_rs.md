# Step 3.1: `oracle/src/dual.rs`, Bilangan Dual untuk Turunan Eksak Tanpa Rumus Baru

> File baru Step 3. Ia memecahkan masalah praktis: Hessian hinge **eksak** butuh turunan kedua sudut lipatan θ, yang rumus
> tertutupnya panjang dan rawan salah. Bilangan dual memberi turunan itu **secara otomatis**, langsung dari rumus gradien yang
> sudah ada dan sudah teruji.

---

## 1. Masalahnya

Energi hinge = `k θ²`. Turunan keduanya:

```
∇²(kθ²) = 2k (∇θ ∇θᵀ + θ ∇²θ)
```

- `∇θ` sudah ada di `hinge.rs` (`angle_gradient`, rumus Bridson, diverifikasi FD sejak Step 1).
- `∇²θ` (12×12 per hinge) belum ada. Step 1 memakai Hessian **Gauss–Newton** `2k∇θ∇θᵀ`, yang membuang suku `θ∇²θ`.
  Itu eksak hanya saat kain datar (θ = 0).

Menurunkan `∇²θ` secara manual berarti puluhan baris aljabar vektor. Setiap salah tanda baru ketahuan lewat tes FD.

## 2. Ide bilangan dual

Bilangan dual adalah pasangan `(v, t)`: **nilai** dan **turunannya** terhadap satu arah. Aturan aritmetikanya adalah aturan kalkulus:

| Operasi | Nilai | Turunan |
|---|---|---|
| a + b | a.v + b.v | a.t + b.t |
| a · b | a.v · b.v | a.t·b.v + a.v·b.t (aturan perkalian) |
| a / b | a.v / b.v | (a.t·b.v − a.v·b.t) / b.v² (aturan pembagian) |
| √a | √a.v | a.t / (2√a.v) |

Kalau kita menjalankan **fungsi apa pun** dengan input dual, dengan input ke-j diberi `t = 1` dan input lain `t = 0`, maka bagian `t` output adalah
**turunan output terhadap input j**, eksak sampai pembulatan. Ini disebut **forward-mode automatic differentiation**.

Analogi backend: seperti *tracing* atau *middleware* yang menempel di setiap operasi. Setiap kali dua angka dijumlah atau dikali, middleware ikut
menghitung "kalau input berubah sedikit, hasil berubah berapa".

## 3. Kode

```rust
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub v: f64,
    pub t: f64,
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual { v: self.v * o.v, t: self.t * o.v + self.v * o.t }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual { v: self.v / o.v, t: (self.t * o.v - self.v * o.t) / (o.v * o.v) }
    }
}
```

Rust mengizinkan **operator overloading** lewat trait `Add`, `Sub`, `Mul`, `Div`, `Neg`. Setelah itu `a * b` pada dua `Dual` otomatis memanggil `mul`.

## 4. Trait `Scalar`: satu fungsi, dua tipe

```rust
pub trait Scalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn constant(value: f64) -> Self;
    fn sqrt(self) -> Self;
}
impl Scalar for f64 { ... }
impl Scalar for Dual { ... }
```

Fungsi yang ditulis generik `fn f<T: Scalar>(...)` bisa dijalankan dengan `f64` (nilai biasa) atau `Dual` (nilai + turunan). `constant` membuat
konstanta (turunan nol), misalnya `T::constant(1.0)`.

Plus fungsi vektor generik: `sub3`, `add3`, `scale3`, `dot3`, `cross3` atas `[T; 3]`.

## 5. Pemakaian di `hinge.rs`

```rust
pub fn angle_hessian(&self, x: &[Vec3]) -> [[Mat3; 4]; 4] {
    let points = self.vertices.map(|v| x[v]);
    let mut blocks = [[[[0.0; 3]; 3]; 4]; 4];
    for (l, c) in (0..4).flat_map(|l| (0..3).map(move |c| (l, c))) {
        let seeded = ... Dual { v: points[k][d], t: 1 kalau (k, d) == (l, c), selain itu 0 } ...;
        let column = angle_gradient_of(seeded);
        ... blocks[k][l][r][c] = column[k][r].t ...
    }
    blocks
}
```

12 lintasan (4 titik × 3 komponen). Setiap lintasan memberi satu **kolom** `∇²θ`. Rumus yang didiferensiasi adalah `angle_gradient_of`, versi generik
dari `angle_gradient` yang sudah teruji. Tidak ada rumus baru yang harus dipercaya.

## 6. Kenapa bukan beda hingga?

Beda hingga `(g(x+h) − g(x−h))/2h` hanya **aproksimasi**, dengan galat pemotongan O(h²) dan galat pembulatan O(ε/h). Akurasi terbaiknya ~10⁻⁸ dalam f64,
dan jauh lebih buruk dalam f32. Bilangan dual eksak sampai pembulatan mesin, dan biayanya sama dengan satu evaluasi gradien per kolom. Beda hingga tetap
dipakai, tetapi sebagai **pemeriksa independen** (tes FD), bukan sebagai metode.

## 7. Tes

```rust
#[test]
fn dual_derivatives_match_calculus() {
    let x = Dual { v: 2.0, t: 1.0 };
    let y = (x * x + Dual::constant(3.0)) / x.sqrt();
    let exact = (2.0 * 2.0 * 2.0f64.sqrt() - (4.0 + 3.0) / (2.0 * 2.0f64.sqrt())) / 2.0;
    assert!((y.t - exact).abs() < 1e-14, ...);
}
```

`y = (x² + 3)/√x`, dengan turunan yang dihitung tangan `y' = (2x√x − (x²+3)/(2√x))/x`. Cocok sampai 10⁻¹⁴.

## 8. Kembaran GPU

WGSL tidak punya generik atau operator overloading. `hinge.wgsl` (dokumen 3.7) menulis aritmetika dual secara manual: struct `D1`/`D3` dan fungsi
`d1_div`, `d3_cross`, dan seterusnya, dengan rumus yang sama persis.

## 9. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Bilangan dual | (nilai, turunan) dengan aturan kalkulus di setiap operasi |
| Forward-mode AD | satu lintasan = satu kolom turunan |
| Trait `Scalar` | satu fungsi generik untuk f64 dan Dual |
| 12 lintasan | 12 kolom ∇²θ per hinge |
| Tanpa rumus baru | yang didiferensiasi adalah gradien yang sudah teruji |
