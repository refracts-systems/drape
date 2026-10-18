# Step 5.9: `gpu/kernels/ccd.wgsl`, Batas Langkah Line Search di GPU

> CCD (*continuous collision detection*) di sini sangat sederhana: untuk setiap vertex, cari α terbesar sehingga gerak sepanjang arah Newton p tidak
> menghabiskan lebih dari 90% jaraknya ke collider. GPU menghitung satu batas per baris, dan host mengambil minimumnya.

---

## 1. Kenapa perlu kernel sendiri

Line search Newton menguji α = α₀, α₀/2, α₀/4, …. Di E3/E4, α₀ = 1. Dengan barrier, α₀ harus menjamin tidak ada vertex yang **melompati** permukaan
(dokumen 5.4). α₀ = min atas semua vertex dan collider. Dengan 16.641 vertex (n129) × 2 collider, menghitungnya di GPU lebih murah daripada mengunduh
posisi dan arah ke host di setiap iterasi Newton.

## 2. Kernelnya

```wgsl
override KEEP: f32 = 0.1;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read> row_vertex: array<u32>;
@group(0) @binding(3) var<storage, read> colliders: array<f32>;
@group(0) @binding(4) var<storage, read_write> bounds: array<f32>;
```

- `KEEP` adalah konstanta `override` yang diisi host dari `problem.json` (`contact.keep_fraction`), bukan angka yang ditulis dua kali. Analogi:
  variabel environment yang dibaca saat pipeline dibuat.
- Buffer `direction` sama dengan yang dibaca kernel perubahan energi (arah Newton dalam tata letak per vertex).
- `bounds` punya satu float per baris unknown.

```wgsl
fn collider_step(k: u32, x: vec3f, p: vec3f) -> f32 {
  let d0 = proximity(k, x).distance;
  if (!is_plane(k)) {
    return sphere_step(k, x, p, d0);
  }
  let rate = dot(collider_point(k), p);
  if (rate >= 0.0) {
    return 1.0;
  }
  return min(1.0, (1.0 - KEEP) * d0 / -rate);
}
```

Lantai: d berubah linear dengan laju n·p. Kalau menjauh (laju ≥ 0), tidak ada batas.

```wgsl
fn sphere_step(k: u32, x: vec3f, p: vec3f, d0: f32) -> f32 {
  let q = x - collider_point(k);
  let target_radius = collider_size(k) + KEEP * d0;
  let a = dot(p, p);
  let b = 2.0 * dot(q, p);
  let c = (1.0 - KEEP) * d0 * (length(q) + target_radius);
  let discriminant = b * b - 4.0 * a * c;
  if (b >= 0.0 || discriminant < 0.0) {
    return 1.0;
  }
  return min(1.0, 2.0 * c / (-b + sqrt(discriminant)));
}
```

Rumus yang sama dengan oracle, termasuk bentuk c tanpa cancellation dan akar stabil.

## 3. Host: minimum dan vertex pembatas

```ts
/** The minimum per-row bound of the CCD kernel and the vertex of its row. */
async function maxStep<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<{ readonly alpha: number; readonly vertex: number }> {
  await gpu.backend.run(gpu.ccd);
  const bounds = new Float32Array(await gpu.backend.read(gpu.bounds));
  const row = bounds.reduce((best, value, i) => (value < bounds[best] ? i : best), 0);
  return { alpha: bounds[row], vertex: bounds[row] < 1 ? problem.u32.row_vertex[row] : -1 };
}
```

Minimum di host, bukan reduksi di GPU, sesuai prompt: lebih sederhana, deterministik, dan waktunya dicatat. `run` dihitung di fase **ccd**, `read` di fase
**io** (decorator waktu, dokumen 4.6). Vertex pembatas ikut dicatat untuk laporan.

## 4. Batas presisi f32 yang terukur

Untuk bola, d0 = |q| − r adalah selisih dua angka ≈ 0,125. Galat pembulatan |q| ≈ ε32·0,125 ≈ 7×10⁻⁹ m menjadi galat **relatif** d0 sebesar 7×10⁻⁹/d0,
dan α ikut membawa galat itu. Di d0 = 3×10⁻⁴ m, galatnya ≈ 2,5×10⁻⁵.

Replay oracle (dokumen 5.7) membandingkan α GPU dengan α f64 dari input yang sama, dengan batas prompt 10⁻⁵. Hasilnya di dokumen 5.11. Ini bukan bug di
kernel: f32 tidak bisa melakukan lebih baik untuk |q| − r tanpa mengubah representasi masalahnya.
