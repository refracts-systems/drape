# Step 3.2: `oracle/src/hinge.rs`, Hinge Tekukan: Sudut, Gradien, Hessian Eksak, dan Perubahan Energi

> Hinge (engsel) sudah ada sejak Step 1, tetapi baru di Step 3 kain benar-benar **menekuk** di ruang 3D. Step 3 menambahkan tiga hal:
> Hessian sudut eksak, tangen energi eksak, dan perubahan energi per hinge untuk line search.

---

## 1. Apa itu hinge

Dua segitiga yang berbagi satu rusuk membentuk "engsel". **Sudut dihedral θ** adalah sudut lipatan antara kedua bidang segitiga:

```
        x1
       /  \
     x3 ──── x4     ← rusuk bersama
       \  /
        x2
```

Urutan titik: `[x1 (lawan kiri), x2 (lawan kanan), x3 (awal rusuk), x4 (akhir rusuk)]`. Kain datar → θ = 0. Energi tekuk = `k θ²`, dengan `k` dari
kekakuan tekuk material dan geometri rest (`hinge_stiffness`: `k_b · 3|e|² / (A1 + A2)`).

Kekakuan tekuk kain sangat kecil (2×10⁻⁶ N·m), tetapi tanpa itu kain tidak punya tahanan terhadap kusut, dan di Step 1 kasus `slack` (hanya tekukan) tidak
bisa diselesaikan PCG.

## 2. Sudut

```rust
pub fn angle(&self, x: &[Vec3]) -> f64 {
    let [x1, x2, x3, x4] = self.vertices.map(|v| x[v]);
    let edge = sub(x4, x3);
    let n1 = cross(sub(x1, x3), sub(x1, x4));
    let n2 = cross(sub(x2, x4), sub(x2, x3));
    let (u1, u2) = (scale(n1, 1.0 / norm(n1)), scale(n2, 1.0 / norm(n2)));
    let sine = dot(cross(u1, u2), scale(edge, 1.0 / norm(edge)));
    sine.atan2(dot(u1, u2))
}
```

- `n1`, `n2`: normal kedua segitiga (cross product dua rusuk).
- `u1`, `u2`: normal satuan.
- `cos θ = u1·u2`; `sin θ = (u1 × u2)·ê` (bertanda, diproyeksikan ke arah rusuk).
- `atan2(sin, cos)` memberi θ di (−π, π] dengan tanda yang benar.

Fungsi `atan2` inilah yang kemudian bermasalah di GPU (dokumen 3.7).

## 3. Gradien sudut, versi generik (baru)

```rust
/// `angle_gradient` written once for any scalar (f64 or Dual), same operation order.
pub fn angle_gradient_of<T: Scalar>(p: [V3<T>; 4]) -> [V3<T>; 4] {
    let [x1, x2, x3, x4] = p;
    let edge = sub3(x4, x3);
    let length = dot3(edge, edge).sqrt();
    let n1 = cross3(sub3(x1, x3), sub3(x1, x4));
    let n2 = cross3(sub3(x2, x4), sub3(x2, x3));
    let (a1, a2) = (scale3(n1, T::constant(1.0) / dot3(n1, n1)), scale3(n2, T::constant(1.0) / dot3(n2, n2)));
    let u1 = scale3(a1, length);
    let u2 = scale3(a2, length);
    let u3 = add3(scale3(a1, dot3(sub3(x1, x4), edge) / length), scale3(a2, dot3(sub3(x2, x4), edge) / length));
    let u4 = add3(scale3(a1, -dot3(sub3(x1, x3), edge) / length), scale3(a2, -dot3(sub3(x2, x3), edge) / length));
    [u1, u2, u3, u4].map(|u| scale3(u, T::constant(-1.0)))
}
```

Salinan baris-demi-baris dari `angle_gradient` lama, tetapi generik atas `Scalar` (dokumen 3.1). Fungsi lama **tidak diubah**, karena dipakai E1.
Tes `generic_angle_gradient_matches_the_f64_one` memakai `assert_eq!` (bit identik, bukan "kira-kira"), sehingga terbukti urutan operasinya sama persis.

## 4. Hessian sudut eksak (baru)

```rust
/// Exact Hessian of θ: each column is the forward-mode derivative of `angle_gradient_of` along one coordinate.
pub fn angle_hessian(&self, x: &[Vec3]) -> [[Mat3; 4]; 4] { ... 12 lintasan dual ... }
```

Detail di dokumen 3.1. `f64::from(u8::from(k == l && d == c))` adalah cara tanpa `as` dan tanpa if untuk mengubah boolean menjadi 1.0/0.0.

## 5. Tangen energi eksak (baru)

```rust
/// Exact energy Hessian 2k(∇θ∇θᵀ + θ∇²θ), the tangent Newton needs away from the flat state.
pub fn tangent(&self, x: &[Vec3]) -> [[Mat3; 4]; 4] {
    let (theta, g, h) = (self.angle(x), self.angle_gradient(x), self.angle_hessian(x));
    let factor = 2.0 * self.stiffness;
    std::array::from_fn(|k| std::array::from_fn(|l| std::array::from_fn(|r| std::array::from_fn(|c| factor * (g[k][r] * g[l][c] + theta * h[k][l][r][c])))))
}
```

`std::array::from_fn` empat tingkat membangun array `[4][4][3][3]` dengan closure, setara empat loop bersarang.

| | `hessian` (Step 1, Gauss–Newton) | `tangent` (Step 3, eksak) |
|---|---|---|
| Rumus | 2k ∇θ∇θᵀ | 2k (∇θ∇θᵀ + θ∇²θ) |
| Selalu PSD? | ya | tidak |
| Eksak? | hanya saat θ = 0 | ya |

Di Step 3, tangen eksak inilah yang diminta, tetapi kemudian **diganti** dengan Gauss–Newton karena trigger (dokumen 3.4). Keduanya tetap ada:
eksak untuk cek FD dan cek kernel, Gauss–Newton untuk simulasi.

## 6. Perubahan energi (baru)

```rust
/// k(θ' − θ)(θ' + θ) for x → x + αd: one hinge's energy change, never a difference of global energies.
pub fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
    let moved: Vec<Vec3> = self.vertices.iter().map(|&v| add(x[v], scale(d[v], alpha))).collect();
    let local = HingeElement { vertices: [0, 1, 2, 3], stiffness: self.stiffness };
    let (before, after) = (self.angle(x), local.angle(&moved));
    self.stiffness * (after - before) * (after + before)
}
```

`kθ'² − kθ² = k(θ' − θ)(θ' + θ)`. Bentuk faktor ini dihitung **per hinge**, sehingga tidak ada pengurangan energi global yang besar (pelajaran E2:
pembatalan angka f32). Energi hinge tidak kuadratik dalam posisi, jadi tidak ada identitas eksak seperti StVK. Tetapi per hinge, selisih θ adalah angka
O(langkah) yang dihitung dari dua sudut O(0,1), dan itu aman.

Trik `local`: titik-titik yang digeser dikumpulkan ke vektor kecil 4 elemen, lalu hinge sementara dengan indeks `[0,1,2,3]` membaca darinya. Tidak perlu
menyalin seluruh array posisi.

## 7. Tes (baru di Step 3)

| Tes | Memeriksa |
|---|---|
| `generic_angle_gradient_matches_the_f64_one` | versi generik bit-identik dengan versi lama |
| `exact_tangent_matches_gradient_finite_difference_when_bent` | tangen eksak = turunan FD gradien, pada hinge yang **tertekuk** (θ ≠ 0), kondisi di mana Gauss–Newton salah |
| `angle_hessian_is_symmetric` | ∇²θ simetris (sifat wajib Hessian) |
| `energy_change_equals_energy_difference` | identitas perubahan energi |

## 8. Ringkasan

| Fungsi | Status | Dipakai untuk |
|---|---|---|
| `angle` | lama | energi, θ |
| `angle_gradient` | lama, tidak diubah | gaya (E1, E3) |
| `angle_gradient_of` | baru, generik | sumber diferensiasi dual |
| `angle_hessian` | baru | ∇²θ eksak |
| `tangent` | baru | Hessian energi eksak |
| `hessian` | lama (GN) | tangen proyeksi setelah trigger |
| `energy_change` | baru | line search Armijo |
