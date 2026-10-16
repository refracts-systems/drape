# Step 3.3: `oracle/src/newton.rs`, Newton Menjadi Generik Lewat Trait

> Di Step 2, `newton.rs` hanya bisa menyelesaikan `Coupon`. Step 3 butuh Newton yang sama untuk satu langkah waktu drape. Daripada menyalin
> loop Newton, file ini diubah menjadi **generik** atas sebuah trait. Perilaku E2 tidak berubah sedikit pun.

---

## 1. Trait `NewtonSystem`

```rust
/// What Newton needs from a discretised energy: residual and tangent on the unknowns, a cancellation-free
/// energy change, and the map between unknown rows and vertices.
pub trait NewtonSystem {
    fn residual(&self, x: &[Vec3]) -> Vec<f64>;
    fn tangent(&self, x: &[Vec3]) -> Bsr;
    fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64;
    fn scatter(&self, rows: &[f64]) -> Vec<Vec3>;
    fn gather(&self, vertices: &[Vec3]) -> Vec<f64>;
}
```

Lima kemampuan yang dibutuhkan Newton dari "sebuah soal energi":

| Metode | Untuk |
|---|---|
| `residual` | gaya tak seimbang pada unknown (g) |
| `tangent` | matriks untuk `K d = −g` |
| `energy_change` | line search Armijo |
| `scatter` | vektor per-baris → per-titik (arah Newton) |
| `gather` | vektor per-titik → per-baris (untuk `gᵀd`) |

Analogi backend: ini **interface** seperti `Repository` atau `PaymentProvider`. Algoritma (Newton) ditulis sekali terhadap interface, lalu setiap soal
(kupon, drape) menyediakan implementasinya.

## 2. Coupon mengimplementasikannya

```rust
impl NewtonSystem for Coupon {
    fn residual(&self, x: &[Vec3]) -> Vec<f64> {
        self.free_gradient(x)
    }
    fn tangent(&self, x: &[Vec3]) -> Bsr {
        Coupon::tangent(self, x)
    }
    ...
}
```

`Coupon::tangent(self, x)` (sintaks panggilan penuh) memanggil metode **bawaan** Coupon yang bernama sama, bukan metode trait. Ini menghindari ambiguitas
dan rekursi tak sengaja.

## 3. Newton generik

```rust
pub fn solve_step<S: NewtonSystem>(system: &S, start: Vec<Vec3>, scale: f64, settings: &NewtonSettings) -> Result<StepSolution, String> {
    ...
        let gradient = system.residual(&step.positions);
    ...
        let direction = newton_direction(system, &step.positions, &gradient)?;
        let Some(search) = line_search(system, &step.positions, &gradient, &direction, settings) else { ... };
    ...
}

pub fn line_search<S: NewtonSystem>(system: &S, ...) -> Option<LineSearch> {
    let slope = free_dot(gradient, &system.gather(direction));
    ...
        .find(|s| system.energy_change(x, direction, s.step_length) <= settings.armijo * s.step_length * slope)
}
```

`<S: NewtonSystem>` adalah **generik dengan trait bound**: kompiler membuat salinan khusus `solve_step` untuk `Coupon` dan untuk `DrapeStep`
(*monomorphization*), sehingga tidak ada biaya runtime dibanding kode yang ditulis khusus.

Isi algoritma (loop, Cholesky, Armijo) **tidak berubah satu karakter pun**. Hanya `coupon.xxx` diganti `system.xxx`.

## 4. Bukti tidak ada perubahan perilaku

- Semua tes E2 (`armijo_backtracks_an_overlong_newton_step`, patch, FD, bolak-balik) tetap lulus.
- Regresi end-to-end: `uniaxial-n33` dijalankan ulang, dan `result.json`-nya **identik** dengan versi commit E2, kecuali waktu GPU.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Trait | kontrak 5 metode yang dibutuhkan Newton |
| Generik + bound | satu loop Newton untuk kupon dan drape |
| Monomorphization | tanpa biaya runtime |
| Refactor aman | isi algoritma identik; regresi E2 identik |
