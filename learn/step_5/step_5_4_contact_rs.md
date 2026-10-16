# Step 5.4: `oracle/src/contact.rs`, Collider Analitik dan Barrier IPC

> Fisika baru pertama sejak Step 3: kain boleh **menyentuh** benda tegar, tetapi tidak boleh **menembusnya**. File ini memuat seluruh model kontak di
> oracle f64: jarak ke collider, fungsi barrier dan turunannya, serta aturan langkah maksimum untuk line search.

---

## 1. Collider analitik

```rust
pub enum Collider {
    Plane { normal: Vec3, offset: f64 },
    Sphere { centre: Vec3, radius: f64 },
}
```

"Analitik" berarti bentuknya diberikan sebagai rumus, bukan sebagai mesh segitiga:

| Collider | Jarak bertanda d(x) | ∇d (normal) | ∇²d |
|---|---|---|---|
| Lantai | n·x − offset (= z) | n | 0 |
| Bola | \|x − c\| − r | (x − c)/\|x − c\| | (I − n nᵀ)/\|x − c\| |

d > 0 di luar, d = 0 tepat di permukaan, d < 0 di dalam. Kontak hanya dihitung antara **vertex** kain dan collider: tepi dan muka segitiga diabaikan
(di luar cakupan prompt).

Analogi backend: `enum` Rust mirip tagged union atau `sealed class`. `match` memaksa kita menangani setiap varian, sehingga menambah collider ketiga akan
menghasilkan error kompilasi di semua tempat yang lupa menanganinya.

## 2. Barrier IPC

Daripada melarang penetrasi sebagai **constraint** (rumit untuk Newton), kontak dimodelkan sebagai **energi** yang menuju tak hingga saat d → 0:

```
b(d) = −κ (d − d̂)² ln(d/d̂)   untuk 0 < d < d̂,   dan 0 untuk d ≥ d̂
```

- **d̂** (activation distance) = 10⁻³·L = 0,5 mm. Di luar jarak ini barrier tidak ada sama sekali, jadi tidak ada gaya "dari jauh".
- Di d = d̂ nilai, turunan pertama, dan turunan kedua sama-sama 0, sehingga energi mulus (C²). Newton butuh ini.
- Saat d → 0, ln(d/d̂) → −∞, jadi b → +∞. Minimisasi Φ **tidak akan pernah** memilih titik yang menembus, selama line search tidak melompatinya
  (bagian 4).

Turunannya, yang dicek dengan beda hingga di tes `barrier_derivatives_match_finite_differences`:

```rust
pub fn first(&self, d: f64) -> f64 {   // b′(d)
    ...
    -self.kappa * (2.0 * t * (d / self.dhat).ln() + t * t / d)        // t = d − d̂
}

pub fn second(&self, d: f64) -> f64 {  // b″(d)
    ...
    -self.kappa * (2.0 * (d / self.dhat).ln() + 4.0 * t / d - t * t / (d * d))
}
```

Analogi backend: barrier mirip **back-pressure** yang makin keras saat antrean mendekati penuh. Tidak ada penolakan tiba-tiba di batas; tekanannya naik
mulus dan tak terbatas tepat di batas.

## 3. κ yang tidak adaptif

```rust
/// κ for which a vertex of mass `mass` under gravity `gravity` rests at d = d̂/2: κ = m·g / |b′(d̂/2)/κ|.
pub fn resting_kappa(dhat: f64, mass: f64, gravity: f64) -> f64 {
    let unit = Barrier { dhat, kappa: 1.0 };
    mass * gravity / unit.first(0.5 * dhat).abs()
}
```

IPC asli menyesuaikan κ selama simulasi. Prompt memilih κ **tetap** dengan arti fisik yang jelas: vertex terberat yang diam di atas collider (hanya
gravitasi yang menekannya) akan seimbang tepat di d = d̂/2. Karena b′ linear terhadap κ, cukup hitung b′(d̂/2) dengan κ = 1 lalu bagi.

## 4. Langkah maksimum (α_max)

```rust
/// Largest α ∈ (0, 1] with d(x + αp) ≥ 0.1·d(x).
pub fn max_step(&self, x: Vec3, p: Vec3) -> f64
```

Line search Newton akan menguji x + αp. Kalau α terlalu besar, vertex bisa **melompati** permukaan: d berubah dari positif ke negatif tanpa pernah
"merasakan" barrier. Karena itu α dibatasi supaya setiap vertex mempertahankan minimal 10% dari jaraknya sekarang (`KEEP_FRACTION = 0.1`).

**Lantai:** d(α) = d0 + α·(n·p) linear, jadi akarnya langsung α = 0,9·d0/(−n·p) saat mendekat (n·p < 0).

**Bola:** |q + αp| = r + 0,1·d0 dengan q = x − c. Dikuadratkan menjadi a·α² + b·α + c = 0 dengan:

```rust
let (a, b) = (dot(p, p), 2.0 * dot(q, p));
let target = radius + KEEP_FRACTION * d0;
let c = (1.0 - KEEP_FRACTION) * d0 * (length(q) + target);   // = |q|² − target², tanpa pengurangan dua angka besar
...
(2.0 * c / (-b + discriminant.sqrt())).min(1.0)
```

Dua trik numerik:

1. **c tanpa cancellation.** |q|² − (r + 0,1·d0)² adalah selisih dua angka hampir sama (~0,0156). Ditulis ulang sebagai (|q| − target)(|q| + target)
   = 0,9·d0·(|q| + target), sehingga tidak ada pengurangan dua angka besar.
2. **Akar kecil dalam bentuk stabil.** Rumus abc biasa (−b − √D)/2a juga baik di sini karena b < 0, tetapi bentuk 2c/(−b + √D) adalah bentuk standar
   yang tidak pernah mengurangkan dua angka hampir sama.

Kalau b ≥ 0 (vertex menjauh) atau D < 0 (lintasan tidak memotong bola), α = 1.

GPU memakai rumus yang sama dalam f32 (`ccd.wgsl`, dokumen 5.9), dan oracle memutar ulang hasilnya (dokumen 5.7).

## 5. `Contact`: semua collider, satu barrier

```rust
pub struct Contact {
    pub colliders: Vec<Collider>,
    pub barrier: Barrier,
}
```

| Metode | Rumus |
|---|---|
| `energy(x)` | Σ_k b(d_k) |
| `gradient(x)` | Σ_k b′(d_k)·n_k |
| `hessian(x)` | Σ_k b″ n nᵀ + b′·(I − n nᵀ)/\|x − c\| |
| `max_step(x, p)` | min_k α_k |
| `min_distance(x)` | min_k d_k |

Hessian punya dua suku. b″ n nᵀ positif: barrier makin kaku ke arah normal. Suku kedua, b′∇²d, **negatif semi-definit** di bola, karena b′ < 0 (gaya
mendorong keluar) dan ∇²d positif. Suku inilah yang membuat tangen eksak bisa indefinit dan perlu regularisasi μ dari E4. Prompt menyiapkan trigger
berbasis bukti (buang suku itu kalau μ mencapai batas). Trigger itu tidak terpicu (dokumen 5.11).

Hessian kontak hanya mengisi **blok diagonal 3×3** per vertex, karena kontak vertex–collider tidak menghubungkan dua vertex kain. Pola matriks BSR tidak
berubah, dan preconditioner block-Jacobi menangkap suku kontak secara utuh.

`Contact::none()` (tanpa collider) dipakai oleh E3/E4: semua jumlah kosong, dan setiap jalur kontak di `drape.rs` dilewati dengan `has_contact()`.

## 6. Tes

| Tes | Yang dikunci |
|---|---|
| `barrier_derivatives_match_finite_differences` | b′ dan b″ di 4 jarak antara 0,06·d̂ dan 0,98·d̂ |
| `resting_kappa_balances_gravity_at_half_dhat` | b′(d̂/2) = −m·g |
| `contact_gradient_and_hessian_match_finite_differences` | gradien dan Hessian gabungan lantai + bola |
| `max_step_keeps_a_tenth_of_the_distance` | lantai dan bola: tepat 10% jarak tersisa; menjauh memberi α = 1 |
