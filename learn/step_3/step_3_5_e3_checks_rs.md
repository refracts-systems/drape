# Step 3.5: `oracle/src/e3_checks.rs`, Cek Beda Hingga untuk Φ dan Riwayat Energi

> Versi Step 3 dari `e2_checks.rs`. Bedanya, yang diuji adalah **potensial inkremental Φ** (inersia + membran + hinge) di keadaan 3D yang
> tertekuk, dengan tangen **eksak**, termasuk Hessian hinge hasil bilangan dual.

---

## 1. Keadaan uji: kain terlipat

```rust
/// A smooth fold plus noise: hinges are bent (θ ≠ 0) and membranes stretched and compressed.
pub fn bent_state(drape: &Drape, rng: &mut Lcg) -> Vec<Vec3> {
    let fold = |p: [f64; 2]| 0.05 * (π x / w).sin() * (2π y / h).cos();
    ... [x + derau, y + derau, fold + derau] ...
}
```

Lipatan sinus setinggi 5 cm plus derau acak 10% jarak grid. Yang dibutuhkan:

- hinge **tertekuk** (θ ≠ 0), karena di θ = 0 Hessian eksak dan Gauss–Newton sama, sehingga cek tidak membedakan keduanya;
- membran teregang di sebagian tempat dan terkompresi di tempat lain.

Fungsi yang sama (dengan seed berbeda) menghasilkan `kernel_state` untuk cek kernel WGSL.

## 2. Empat cek

```rust
pub fn drape_derivative_check(columns: usize) -> DrapeDerivativeCheck {
    let drape = Drape::build(columns, TangentKind::Exact);
    ...
    DrapeDerivativeCheck {
        gradient_error: worst(&|v| gradient_error(&step, &x, v)),
        hessian_error: worst(&|v| hessian_error(&step, &x, v)),
        hinge_energy_change_error: worst(&|v| hinge_change_error(&drape, &x, v)),
        energy_change_error: worst(&|v| total_change_error(&step, &x, v)),
    }
}
```

| Cek | Rumus | Gate |
|---|---|---|
| Gradien Φ | `(Φ(x+hv) − Φ(x−hv))/2h` vs `residual·v` | ≤ 10⁻⁶ |
| Hessian Φ (eksak) | `(r(x+hv) − r(x−hv))/2h` vs `K v` | ≤ 10⁻⁵ |
| Identitas hinge | `Σ energy_change` vs `Σ(E(x+αv) − E(x))` | ≤ 10⁻¹² |
| Identitas total | `DrapeStep::energy_change` vs `Φ(x+αv) − Φ(x)` | (dilaporkan) |

Drape dibangun dengan `TangentKind::Exact`, bukan `Projected`. Yang diuji adalah apakah **turunan yang diklaim eksak memang eksak**. Tangen proyeksi
memang sengaja bukan turunan Φ, sehingga cek FD-nya akan "gagal" secara desain.

`DrapeStep::value` (Φ itu sendiri) hanya ada untuk cek ini. Newton tidak pernah memakainya, karena Newton memakai `energy_change`.

### Hasil n33

| Cek | Nilai |
|---|---|
| gradien | 6,0×10⁻⁹ |
| Hessian eksak | 2,7×10⁻¹⁰ |
| identitas hinge | 2,0×10⁻¹⁶ |
| identitas total | 1,1×10⁻¹⁴ |

## 3. Riwayat energi

```rust
/// Total energy per state and the largest per-step increase relative to max |E| (E₀ = 0 for the flat sheet at rest).
pub fn energy_history(drape: &Drape, positions: &[Vec<Vec3>], velocities: &[Vec<Vec3>]) -> EnergyHistory {
    let energies = ...total_energy setiap keadaan...;
    let scale = energies.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    let worst_increase = energies.windows(2).map(|w| (w[1] - w[0]) / scale).fold(f64::NEG_INFINITY, f64::max);
    ...
}
```

Prompt meminta "kenaikan f32 per langkah ≤ 10⁻⁶ **dari energi total**". Energi total awal = 0 (kain datar di z = 0, diam, tanpa regangan), jadi "relatif
terhadap energi saat itu" tidak terdefinisi di langkah awal. Interpretasi yang ditetapkan **sebelum mengukur**: relatif terhadap **max |E|** sepanjang
lintasan. Untuk f64 ("selalu tidak naik") diberi kelonggaran pembulatan 10⁻¹². Keduanya tertulis di `e3_compare.rs`.

`worst_increase` negatif berarti energi **selalu turun**.

## 4. Kecepatan dari posisi

```rust
/// Backward-Euler velocities (x_{n+1} − x_n)/h of a stored trajectory, v₀ = 0.
pub fn velocities_of(positions: &[Vec<Vec3>], time_step: f64) -> Vec<Vec<Vec3>> { ... }
```

GPU hanya menyimpan posisi. Kecepatan backward Euler diturunkan dengan rumus yang sama persis dengan oracle, jadi tidak perlu disimpan.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Keadaan terlipat | hinge tertekuk supaya Hessian eksak benar-benar diuji |
| Tangen eksak untuk cek | proyeksi memang bukan turunan Φ |
| Energi relatif max |E| | karena E₀ = 0 |
| Kecepatan diturunkan | dari posisi, rumus identik oracle |
