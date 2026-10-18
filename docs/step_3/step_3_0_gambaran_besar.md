# Step 3.0: Gambaran Besar Experiment 3, Kain yang Bergerak dalam 3D

> Step 1 menguji solver linear. Step 2 menguji Newton untuk kain yang ditarik di bidang datar. Step 3 menggabungkan semuanya menjadi
> **simulasi kain pertama yang benar-benar seperti kain**: tiga dimensi, bisa menekuk, bergerak dalam waktu, dan berkerut.

---

## 1. Pertanyaan Step 3

> Apakah rantai GPU f32 (kernel fisika → Newton → PCG) tetap **benar** dan **stabil** ketika fisikanya penuh: 3D, nonlinear, dinamis,
> dan berkerut? Dan seberapa cepat?

## 2. Benchmark: kain yang dijatuhkan

- Lembar kain rajut 0,5 m × 0,5 m (material yang sama dengan Step 1–2, arah wale = x).
- Mula-mula **datar horizontal** dan diam.
- **Dua sudut atas dipaku**; sisanya bebas di ketiga arah.
- Dilepas, lalu gravitasi menariknya ke bawah. Kain berayun turun sampai menggantung, terkompresi di tengah, berkerut, dan bergoyang.
- **240 langkah** backward Euler dengan h = 1/240 detik (1 detik simulasi).
- Mesh n33 dan n65 (dibandingkan dengan oracle f64), n129 (GPU saja, untuk waktu dan skala).

Kondisi ini sengaja sulit: kompresi dan kerutan adalah kondisi numerik terberat untuk kain.

## 3. Apa yang baru dibanding Step 2

| Komponen | Step 2 | Step 3 |
|---|---|---|
| Dimensi | 2D (z dikunci) | 3D penuh |
| Tekukan | tidak ada | hinge (sudut lipatan antar segitiga) |
| Waktu | langkah beban statis | backward Euler 240 langkah |
| Energi per langkah | Π (elastis) | Φ = inersia + membran + hinge |
| Hessian hinge | – | eksak lewat bilangan dual |
| Solver linear | E1 PCG | E1 PCG (tidak diubah) |
| Kernel perakitan | E2 | E2 (tidak diubah) |

## 4. Konsep fisika baru dalam bahasa backend

### Backward Euler sebagai minimisasi

Setiap langkah waktu dicari dengan meminimalkan:

```
Φ(x) = 1/(2h²)(x − x̂)ᵀM(x − x̂) + Π_membran(x) + Π_hinge(x)
x̂   = x_n + h·v_n + h²·g
```

Analogi: x̂ adalah "ke mana setiap titik akan pergi kalau tidak ada kain yang menahan" (meneruskan kecepatan + jatuh bebas). Suku pertama menghukum penyimpangan
dari situ (inersia), dan sisanya menghukum peregangan dan tekukan. Jawabannya kompromi terbaik. Karena berbentuk minimisasi, Newton + Armijo dari Step 2 dipakai
apa adanya.

### Hinge (tekukan)

Dua segitiga yang berbagi rusuk membentuk engsel dengan sudut lipatan θ. Energinya `kθ²`. Kekakuannya kecil (kain mudah dilipat), tetapi tanpa hinge kain tidak
punya tahanan terhadap lipatan.

### Bilangan dual (dokumen 3.1)

Cara mendapatkan turunan kedua yang eksak secara otomatis dari rumus turunan pertama yang sudah teruji, tanpa menurunkan rumus baru dengan tangan.

### Konsistensi satu langkah

Kain berkerut bersifat **chaos**: perbedaan pembulatan sekecil apa pun membuat kerutan muncul di tempat berbeda setelah ratusan langkah. Karena itu perbandingan
penentu bukan "lintasan f32 vs f64 di akhir", melainkan: mulai dari **keadaan f64 yang sama**, jalankan **satu** langkah GPU, lalu bandingkan dengan langkah f64.
Itu mengukur galat solver, bukan sensitivitas fisika.

## 5. Dua kejadian penting

### Trigger: tangen eksak kehilangan positif definit (dokumen 3.4)

Di langkah 87 (n33), saat kain terkompresi, band Cholesky oracle menolak tangen eksak (`pivot -1.21e-1`). Aturan yang sudah ditetapkan di prompt langsung berlaku:
oracle dan GPU beralih ke **Hessian membran terjepit + Hessian hinge Gauss–Newton**. Efeknya terukur: Newton menjadi linear (tidak lagi kuadratik) dan banyak
langkah mentok di 30 iterasi.

### Bug presisi: `atan2` WGSL hanya 4096 ULP (dokumen 3.7)

Gate kernel menangkap galat 5×10⁻⁵ di gaya hinge. Penyebabnya, fungsi bawaan `atan2` WGSL boleh salah sampai ~2,4×10⁻⁴ relatif. Diganti dengan `atanf` Cephes
(~2 ULP), sehingga galat turun ke 1,7×10⁻⁷. Gate tidak diubah.

## 6. Hasil singkat (detail di dokumen 3.19)

| | n33 | n65 | n129 |
|---|---|---|---|
| Konsistensi satu langkah (gate ≤ 10⁻³) | ≤ 4,7×10⁻⁵ ✓ | ≤ 4,6×10⁻⁵ ✓ | – |
| Lantai f32 = lantai representasi | ✓ | ✓ | – |
| Energi tidak naik | ✓ | ✓ | ✓ (f32) |
| Kernel WGSL vs f64 | ≤ 2,7×10⁻⁷ ✓ | ≤ 2,6×10⁻⁷ ✓ | ≤ 2,5×10⁻⁷ ✓ |
| f64 konvergen atau di lantai | 197/240 ✗ | 170/240 ✗ | – |
| ms per langkah (anggaran 4,17) | 903 | 1742 | 7180 |

**Kesimpulan satu kalimat**: GPU f32 cukup **akurat** dan **stabil** untuk kain dinamis yang berkerut. Hambatannya adalah **konvergensi Newton** (setelah trigger)
dan **kecepatan**, yang ratusan kali di atas anggaran real-time.

## 7. Cara menjalankan

```bash
./run-e3.sh hardware                       # n33, n65 (dengan oracle) + n129 (GPU saja)
SIZES="33" GPU_ONLY="" ./run-e3.sh hardware
```

Oracle n65 butuh sekitar 72 menit (Cholesky ribuan kali); GPU n129 sekitar 30 menit.
