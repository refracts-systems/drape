# Step 1.6: `gpu/kernels/state.wgsl`, Dasar WGSL Bagian 1 (Struct, Buffer, Tipe, Layout Memori)

> File ini hanya berisi satu `struct`. Tetapi ia pintu masuk yang bagus untuk WGSL: tipe data, struct,
> bagaimana data hidup di memori GPU, dan bagaimana host (TypeScript) dan GPU sepakat soal byte.

---

## 1. Isi file

```wgsl
// Shared CG scalars, prepended by the host to every kernel that reads or writes them.
// replace_above2: residual replacement runs only while rr > replace_above2 · rr0.
// status: 0 running, 1 converged, 2 iteration limit, 3 breakdown (pᵀq ≤ 0 or not finite).
struct State {
  rz: f32, pq: f32, alpha: f32, beta: f32,
  rr: f32, rr0: f32, tol2: f32, true_rr: f32,
  iteration: u32, status: u32, max_iterations: u32, replace_above2: f32,
  pad1: u32, pad2: u32, pad3: u32, pad4: u32,
}
```

## 2. WGSL itu apa, dalam satu paragraf

WGSL (*WebGPU Shading Language*) adalah bahasa untuk menulis program yang dijalankan GPU. Sintaksnya mirip Rust
(`let`, `var`, `fn`, `struct`, tipe setelah titik dua), tetapi jauh lebih kecil. Tidak ada heap, tidak ada string,
tidak ada rekursi, tidak ada pointer bebas, dan tidak ada `print`. Semua ukuran array di memori lokal harus diketahui saat kompilasi.
Program WGSL dikompilasi oleh browser/driver menjadi kode mesin GPU (di Windows lewat D3D12 → DXIL atau DXBC).

Untuk backend developer, cara berpikir yang paling membantu:

> Satu file WGSL = satu **handler** yang dipanggil puluhan ribu kali secara paralel. Handler tidak punya state sendiri
> di antara pemanggilan; ia hanya membaca dan menulis **buffer** yang disiapkan host.

## 3. Tipe data dasar

| WGSL | Arti | Padanan |
|---|---|---|
| `f32` | float 32-bit | `float` / `number` (tapi 32-bit) |
| `u32` | unsigned int 32-bit | `uint32` |
| `i32` | signed int 32-bit | `int32` |
| `bool` | boolean | `boolean` |
| `vec3f` | 3 × f32 | `[number, number, number]` |
| `vec3u` | 3 × u32 | tiga indeks |
| `array<f32>` | array f32 yang panjangnya ditentukan saat runtime | `Float32Array` |
| `array<f32, 256>` | array dengan panjang tetap | array C |

**Tidak ada f64.** WebGPU tidak mendukung angka 64-bit. Ini akar seluruh pertanyaan Experiment 1.

Literal angka: `0u` adalah u32, `1.0` adalah f32 (atau "abstract float" yang menyesuaikan), `256u` adalah u32.
WGSL **tidak** mengonversi tipe secara otomatis: `u32 + i32` adalah error kompilasi. Konversi harus eksplisit: `f32(n)`, `u32(x)`.

## 4. Struct `State`: "baris konfigurasi" bersama

CG punya beberapa skalar yang harus dibagi antar kernel: `rz`, `α`, `β`, jumlah iterasi, status. Di CPU ini sekadar variabel lokal.
Di GPU, setiap kernel adalah program terpisah yang dipanggil terpisah. Satu-satunya cara berbagi state adalah lewat **buffer**.

Maka semua skalar dikumpulkan dalam satu struct yang hidup di satu buffer kecil (64 byte). Setiap kernel yang butuh
mengikatnya sebagai:

```wgsl
@group(0) @binding(6) var<storage, read_write> state: State;
```

Analogi: `State` adalah **satu baris di tabel konfigurasi** yang dibaca dan diperbarui oleh beberapa job berbeda secara berurutan.

| Field | Isi | Siapa yang menulis |
|---|---|---|
| `rz` | rᵀz saat ini | reduce MODE 0, 2, 4 |
| `pq` | pᵀq | reduce MODE 1 |
| `alpha`, `beta` | panjang langkah, koefisien arah | reduce MODE 1, 2 |
| `rr`, `rr0` | ‖r‖² sekarang dan awal | reduce MODE 0, 2, 4 |
| `tol2` | toleransi² | **host** (TypeScript) sebelum solve |
| `true_rr` | ‖b − Ax‖² dihitung GPU | reduce MODE 3 |
| `iteration` | jumlah iterasi | reduce MODE 0, 2 |
| `status` | 0 jalan, 1 konvergen, 2 batas iterasi, 3 breakdown | reduce |
| `max_iterations` | batas | **host** |
| `replace_above2` | gate replacement² | **host** (baru di sesi ini, menggantikan `pad0`) |
| `pad1..pad4` | pengisi | tidak dipakai |

## 5. Kenapa ada `pad`? Layout memori

GPU punya aturan **alignment** yang ketat untuk buffer: struct di storage buffer dibulatkan ukurannya ke kelipatan
alignment terbesar anggotanya, dan banyak driver lebih nyaman dengan ukuran kelipatan 16 byte. Dengan 16 field × 4 byte = 64 byte,
struct ini berukuran rapi dan **offset tiap field mudah dihitung**:

```
offset  0: rz        offset 16: rr        offset 32: iteration      offset 48: pad1
offset  4: pq        offset 20: rr0       offset 36: status         ...
offset  8: alpha     offset 24: tol2      offset 40: max_iterations
offset 12: beta      offset 28: true_rr   offset 44: replace_above2
```

Host menulis dan membaca buffer ini sebagai **byte mentah**. Di `solver.ts`:

```ts
function initialState(settings: SolveSettings, replaceAbove: number): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(16);
  const floats = new Float32Array(words.buffer);
  floats[6] = settings.tolerance * settings.tolerance;   // tol2 di offset 24 = indeks 6
  floats[11] = replaceAbove * replaceAbove;                // replace_above2 di offset 44 = indeks 11
  words[10] = settings.maxIterations;                      // max_iterations di offset 40 = indeks 10
  return words;
}
```

Trik penting: `Uint32Array` dan `Float32Array` **berbagi `ArrayBuffer` yang sama**. Menulis `floats[6]` menaruh bit float di word ke-6;
menulis `words[10]` menaruh bit integer di word ke-10. Inilah cara host mengisi struct campuran f32/u32 tanpa library serialisasi.

Membaca kembali:

```ts
function decodeState(bytes: ArrayBuffer): StateSnapshot {
  const floats = new Float32Array(bytes);
  const words = new Uint32Array(bytes);
  return { rr: floats[4], rr0: floats[5], trueRr: floats[7], iteration: words[8], status: words[9] };
}
```

**Jebakan**: kalau urutan field di WGSL diubah tanpa mengubah indeks di TypeScript, tidak ada error kompilasi; angkanya
diam-diam salah. Karena itu saat `pad0` diganti `replace_above2`, posisinya sengaja dipertahankan di indeks 11.

## 6. "Prepended by the host": tidak ada `#include`

WGSL tidak punya sistem modul atau `#include`. Jadi bagaimana tujuh kernel memakai struct yang sama?
Host (`kernels.ts`) **menempelkan teks** `state.wgsl` di depan teks kernel sebelum dikompilasi:

```ts
return [name, header === null ? body : `${await readText(header)}\n${body}`] as const;
```

Ini persis cara kerja preprocessor C, hanya dilakukan manual dengan string concatenation. Dokumen 1.14 membahasnya lebih jauh.

## 7. Address space: di mana variabel hidup

WGSL membedakan tempat tinggal variabel. Ini konsep baru yang paling penting:

| Address space | Hidup di | Dibagi oleh | Kecepatan | Contoh di proyek |
|---|---|---|---|---|
| `function` | register/stack thread | satu thread | tercepat | `var a0 = 0.0;` di dalam fungsi |
| `private` | per thread, global modul | satu thread | cepat | (tidak dipakai) |
| `workgroup` | memori on-chip | 256 thread satu workgroup | sangat cepat, kecil (~16–48 KB) | `var<workgroup> lane: array<f32, 256>;` |
| `storage` | VRAM | **semua** thread, dan host | lambat (relatif) tetapi besar | `var<storage, read> vals: array<f32>;` |
| `uniform` | VRAM, cache khusus | semua thread, read-only | cepat, kecil | (tidak dipakai) |

`State` hidup di `storage` dengan akses `read_write`. Semua thread di semua workgroup melihat buffer yang sama.

## 8. `read` vs `read_write`

```wgsl
var<storage, read> vals: array<f32>;
var<storage, read_write> state: State;
```

- `read`: kernel hanya membaca. Kompiler bisa mengoptimalkan, dan driver tahu tidak ada konflik tulis.
- `read_write`: kernel boleh menulis.

Deklarasi ini harus **cocok** dengan cara host membuat buffer. Di vgpu, buffer dibuat dengan `access: "read"` atau `"read-write"`,
dan sesi kerja membuktikan lewat probe bahwa vgpu **menolak** buffer `read` diikat ke variabel `read_write`, tetapi
**menerima** buffer `read-write` diikat ke variabel `read`. Fakta ini dipakai di Step 2 (dokumen 2.13).

## 9. Kenapa state skalar ada di GPU, bukan di host?

Alternatifnya: setiap iterasi, host membaca `pᵀq` dari GPU, menghitung α di JavaScript, lalu menulis α kembali. Itu berarti
**bolak-balik CPU↔GPU tiga kali per iterasi**, masing-masing puluhan sampai ratusan mikrodetik. Untuk 1500 iterasi,
latensinya bisa lebih besar daripada komputasinya.

Dengan `State` di GPU, seluruh iterasi berjalan di GPU tanpa campur tangan host. Host hanya membaca `status` setiap
32 iterasi (atau 50 kalau replacement aktif). Analogi: daripada round-trip ke database per baris, kirim **stored procedure**
yang memproses batch, lalu cek hasilnya sekali-sekali.

## 10. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| WGSL | bahasa kecil mirip Rust untuk program GPU, tanpa f64, heap, atau rekursi |
| Struct `State` | 16 word (64 byte) skalar CG bersama, satu buffer storage |
| Layout | host menulis/membaca byte mentah lewat `Uint32Array`/`Float32Array` yang berbagi buffer |
| Address space | function (thread), workgroup (tim 256), storage (semua + host) |
| `read`/`read_write` | harus konsisten dengan cara buffer dibuat di host |
| Kenapa di GPU | menghindari round-trip host per iterasi |
