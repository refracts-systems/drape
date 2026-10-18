# Step 7.3: Pemicu μ-cap, Menjatuhkan Suku Kurvatur Barrier (`inertia.wgsl`, `contact.wgsl`, `contact-host.ts`, `drape.rs`)

> Prompt E7 membawa satu **pemicu berbasis bukti**: bila sebuah langkah yang diterima berakhir di batas regularisasi μ (1e3), suku kurvatur
> b′∇²d dari Hessian barrier dijatuhkan untuk langkah itu. Dokumen ini menjelaskan kenapa suku itu bermasalah dan bagaimana saklarnya dipasang
> tanpa mengubah satu bit pun hasil E5/E6.

---

## 1. Masalah: Hessian barrier bisa tidak definit

Hessian barrier satu simpul terhadap bola:

```
H = b″ · n nᵀ + b′ · (I − n nᵀ) / |x − c|
```

Suku pertama (kurvatur sepanjang normal) selalu ≥ 0. Suku kedua adalah kurvatur geometri bola (∇²d = (I − n nᵀ)/|x − c|); karena b′ < 0 di dalam
zona barrier, ia **negatif**, dan saat simpul sangat dekat bola (d kecil, b′ sangat negatif) bisa lebih besar daripada kekakuan kain di sekitarnya.
Akibatnya Hessian tidak definit, LM menaikkan μ sampai batas 1e3, dan Newton berhenti tanpa kemajuan yang berarti.

Analogi backend: **mengabaikan turunan kedua yang tidak stabil** seperti memakai Gauss-Newton alih-alih Newton penuh. Arahnya sedikit kurang akurat,
tetapi selalu bisa dipakai.

## 2. Saklar di WGSL

Header buffer `colliders` sudah punya satu slot cadangan: `[count, d̂, κ, 0, …]`. Slot ke-3 (`colliders[3]`) kini berarti "jatuhkan suku kurvatur":

```wgsl
let entry = second * n[r] * n[c]
          + first * near.curvature * (1.0 - colliders[3]) * (identity - n[r] * n[c]);
```

Nilainya 0 di semua kasus E3–E6, jadi `(1.0 - 0.0)` adalah perkalian dengan 1 persis dan hasilnya **bit-identik** (terbukti oleh regresi E5/E6 yang
identik byte demi byte). Gradien tidak berubah: yang diubah hanya Hessian, jadi titik kesetimbangan tidak bergeser, hanya jalur menuju ke sana.

`setBarrierCurvature(gpu, problem, drop)` di `contact-host.ts` menulis ulang header itu (menyalin `problem.contact.colliders`, mengubah elemen ke-3).

## 3. Pemicunya di GPU (`drape-e7.ts`)

`attemptWithTrigger`:

1. Jalankan percobaan biasa.
2. Bila ia **diterima** (ada pembaruan) **dan** salah satu catatan iterasinya berstatus `regularization_cap`, anggap itu bukti.
3. Catat percobaan pertama sebagai alasan 4 (`curvature_retry`), pasang saklar, ulangi dari x_k, lalu lepas saklar (`finally`).
4. Langkah yang diterima mencatat `curvatureDropped = true`.

Penilai memperlakukan alasan 4 sebagai penolakan yang sah (`rejection_replay`) dan menghitung `curvature_dropped_steps` terpisah dari `rejected_attempts`.

## 4. Pemicu yang sama di oracle

Oracle punya `Drape::barrier_curvature: Cell<bool>` (default benar) yang dibaca `Contact::hessian_with(x, curvature_term)`. `attempt` di `e7_ptc.rs` mengulang
`inner_solve` dengan `barrier_curvature = false` bila solusi pertama mencapai batas dengan kemajuan.

## 5. Hasil

| Kasus | langkah dijatuhkan | langkah μ-cap yang tersisa |
|---|---|---|
| oracle n33 w1 cold | 2 | 0 |
| oracle n17 w1 warm | 1 | 1 |
| oracle n33 w2 cold | 1 | 1 |
| GPU n129 (beberapa) | sampai 18 | – |

Di dua kasus oracle, mengulang tanpa suku itu **tidak menyembuhkan** batas μ, sehingga cek "nol langkah μ-cap diterima" gagal. Itu temuan yang
dilaporkan, bukan sesuatu yang disamarkan. Di n129 pemicu sering menyala (sampai 18 langkah dalam satu solve), tanda bahwa pada mesh halus jalur PTC
banyak melewati zona barrier yang sangat dekat.
