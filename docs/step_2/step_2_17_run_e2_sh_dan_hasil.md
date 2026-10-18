# Step 2.17: `run-e2.sh` dan Hasil Experiment 2

> Dokumen penutup: cara menjalankan, apa yang terukur, dan bagaimana membaca temuan utama. Ini juga menjawab dua pertanyaan pengguna:
> apakah residual f32 mencapai 10⁻⁶, dan apakah inexact Newton tetap kuadratik.

---

## 1. Skrip

```bash
#!/usr/bin/env bash
# Experiment 2 end to end: f64 Newton oracle per coupon, f32 WGSL Newton + E1 PCG on the GPU (vgpu), score, summarize.
#   ./run-e2.sh [adapter]                       vgpu adapters: auto | hardware | software
#   PATHS="patch uniaxial biaxial" SIZES="33 65" ./run-e2.sh hardware
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:-hardware}"
PATHS="${PATHS:-patch uniaxial biaxial}"
SIZES="${SIZES:-33 65}"
ORACLE=./oracle/target/release/oracle

cargo build --release --manifest-path oracle/Cargo.toml
for path in $PATHS; do
  for n in $SIZES; do
    dir="coupon/$path-n$n"
    [ -f "$dir/reference.bin" ] || "$ORACLE" e2-generate --path "$path" --n "$n" --out "$dir"
    (cd gpu && node src/node/run-e2.ts --case "../$dir" --adapter "$ADAPTER")
    "$ORACLE" e2-compare --dir "$dir" > /dev/null
  done
done
"$ORACLE" e2-summarize --dir coupon --out "results/experiment2-vgpu-$ADAPTER.json"
```

Struktur sama persis dengan `run-e1.sh` (dokumen 1.25): generate (dengan cache) → GPU → compare → summarize. Menjalankannya ulang dari nol menghasilkan
**angka yang identik**, sehingga hasilnya deterministik dan dapat direproduksi.

## 2. Hasil per cek

| # | Cek | Hasil | Angka |
|---|---|---|---|
| 1 | F dan Green strain | ✓ | patch f64: F 3×10⁻¹⁴, E 2×10⁻¹³ |
| 2 | Material StVK | ✓ | identitas energi ≤ 5,7×10⁻¹⁵ |
| 3 | Gaya dan tangen | ✓ | WGSL vs f64: gaya ≤ 5,1×10⁻⁷, tangen ≤ 1,9×10⁻⁷ (batas 10⁻⁵) |
| 4 | Newton + Armijo | ✓ | f64 kuadratik; Armijo diuji paksa di tes unit |
| 5 | Patch test | ✓ | f64 perpindahan 1,4×10⁻¹⁶; f32 2,2×10⁻⁷ ukuran kupon |
| 6 | Cek FD | ✓ | gradien ≤ 8,3×10⁻⁸, tangen ≤ 3,0×10⁻⁹ |
| 7 | Kurva bolak-balik | ✓ | f64 menelusuri ulang sampai 10⁻¹⁴; f32 sampai 3,7–7,9×10⁻⁶ |
| 8a | f32 vs f64: perpindahan, tegangan | ✓ | ≤ 2,2×10⁻⁵ / 7,9×10⁻⁶ (batas 10⁻³) |
| 8b | f32 mencapai 10⁻⁶ | **✗** uniaxial/biaxial, ✓ patch | lantai = lantai representasi |
| 8c | iterasi Newton f32 vs f64 | ✓ | sama di setiap langkah (sampai 10⁻⁶ atau lantai) |

## 3. Kurva tegangan-regangan (n33, f64)

**Uniaxial** (regangan wale → tegangan P_xx N/m):

```
regangan: 0,02  0,04  0,06  0,08  0,10   0,12   0,14   0,16   0,18   0,20
naik    : 20,1  41,5  64,1  88,1  113,3  139,9  167,9  197,2  228,0  260,1
turun   : sama persis (228,0 … 20,1 → 0,0)
```

Kurvanya **mengeras** (kemiringan naik dari ~1000 ke ~1600 N/m): efek nonlinear geometrik StVK dalam Green strain. Turun menelusuri naik: hiperelastik,
tanpa histeresis.

**Biaxial** (15% × 7,5%): P_xx naik ke 121,3 N/m dan P_yy ke 28,0 N/m, lalu kembali ke 0.

## 4. Pertanyaan 1: apakah residual f32 mencapai 10⁻⁶?

**Hanya di patch test** (6,3×10⁻⁷ di n33, 6,5×10⁻⁸ di n65). Di uniaxial dan biaxial, Newton f32 berhenti di lantai:

| | langkah naik | langkah turun, terburuk |
|---|---|---|
| n33 | ~5–7×10⁻⁶ | 4,4×10⁻⁵ (uni), 4,7×10⁻⁵ (bi) |
| n65 | ~5×10⁻⁶ | 1,1×10⁻³ (uni), 1,2×10⁻⁴ (bi) |

### Temuan kunci: lantai f32 = lantai representasi

| Kupon | lantai f32 Newton | residual f64 dari iterate f32 | lantai representasi |
|---|---|---|---|
| uniaxial-n33 | 4,43×10⁻⁵ | 4,43×10⁻⁵ | 4,43×10⁻⁵ |
| biaxial-n33 | 4,69×10⁻⁵ | 4,69×10⁻⁵ | 4,63×10⁻⁵ |
| uniaxial-n65 | 1,14×10⁻³ | 1,14×10⁻³ | 1,13×10⁻³ |
| biaxial-n65 | 1,24×10⁻⁴ | 1,24×10⁻⁴ | 1,25×10⁻⁴ |
| patch-n65 | 6,54×10⁻⁸ | 6,54×10⁻⁸ | 6,44×10⁻⁸ |

Cocok di setiap langkah, sampai ~10%. Artinya:

1. Residual yang dilaporkan GPU dalam f32 **jujur**: sama dengan residual sebenarnya dalam f64.
2. Solusi f32 sudah sebaik **solusi f64 yang dibulatkan ke f32**.
3. Tidak ada solver f32 yang bisa lebih baik **selama posisi disimpan sebagai koordinat absolut f32**.

### Kenapa ada lantai?

Koordinat titik sekitar 0,1 m. f32 punya presisi relatif ~6×10⁻⁸, jadi setiap koordinat hanya bisa disimpan dengan langkah ~6 nanometer. Dengan
kekakuan kain, galat posisi sekecil itu sudah menghasilkan gaya tak seimbang. Gaya itu dibandingkan dengan **skala langkah**, yaitu gaya tak seimbang
dari kenaikan beban kecil (2% regangan). Saat skala kecil dibanding gaya dalam yang sudah ada (terutama saat melepas beban dari tegangan tinggi),
rasionya membesar.

f64 punya struktur yang sama, hanya 5×10⁸ kali lebih kecil: di langkah terburuk uniaxial-n65, Newton f64 pun berhenti di 2,1×10⁻¹².
Perbandingannya 1,1×10⁻³ / 2,1×10⁻¹² ≈ 5×10⁸ ≈ ε_f32 / ε_f64. Kedua presisi dibatasi oleh hal yang sama: ketelitian menyimpan posisi.

### Secara fisik

Galat perpindahan ~4×10⁻⁷ dari puncak, dan galat tegangan ~8×10⁻⁶ dari puncak. Untuk simulasi kain, ini jauh lebih teliti daripada yang dibutuhkan mata
maupun pengukuran material. "Residual 10⁻⁶ relatif terhadap kenaikan beban" adalah definisi yang sangat ketat.

## 5. Pertanyaan 2: apakah inexact Newton tetap kuadratik?

**Ya, sampai lantai.** Contoh uniaxial-n33 langkah pertama:

| Iterasi | f64 (Cholesky pasti) | f32 (PCG inexact) |
|---|---|---|
| 0 | 2,045×10⁻³ | 2,046×10⁻³ |
| 1 | 6,41×10⁻⁵ | 6,44×10⁻⁵ |
| 2 | 2,04×10⁻⁸ | 5,07×10⁻⁶ ← lantai |
| 3 | 1,04×10⁻¹⁴ | ~5×10⁻⁶ |

- Sampai lantai, f32 sama dengan f64 dalam ~3 digit.
- η terukur (residual sejati PCG f32) 3×10⁻⁷ sampai 2×10⁻⁵, jauh di bawah forcing term 10⁻². Langkah linear praktis "pasti" untuk Newton.
- Armijo tidak pernah memotong langkah (α = 1 selalu), baik f32 maupun f64.
- Iterasi Newton sampai 10⁻⁶ (atau lantai): f64 2 (uniaxial), 3–4 (biaxial), 7 (patch). f32 mencapai titik yang sama dalam jumlah iterasi yang sama.
- PCG: 100–160 iterasi per solve di n33, 230–320 di n65, kira-kira 2× per penghalusan seperti di E1.

## 6. Perubahan arsitektur yang dipaksa bukti

**Tidak ada.** Lantainya sekarang **terukur dan dijelaskan**. Untuk mencapai 10⁻⁶ dibutuhkan representasi posisi yang lebih teliti dalam f32, tetapi itu
tidak diimplementasikan karena tidak ada gate yang memintanya dan pengguna melarang perubahan arsitektur tanpa bukti yang memaksa.

Keputusan desain yang **dipaksa hasil pengukuran** di dalam lingkup Step 2:

| Bukti | Keputusan |
|---|---|
| Cholesky: tangen pertama tidak positif definit | prediktor afin per langkah |
| Biaxial dengan semua tepi dijepit = patch test | rahang setengah tengah tepi |
| ‖g(x₀)‖ setelah prediktor bisa ≈ derau | skala = gaya tak seimbang dari penjepit |
| vgpu menolak menulis buffer `read` | `vals`/`rhs` solver E1 jadi `read-write` |
| WGSL tidak punya tuple | `let` terpisah |
| Tes patch n9: regangan 4,8×10⁻¹² | batas regangan f64 10⁻¹⁰ (perpindahan tetap 10⁻¹²) |

## 7. Batas lingkup

Sesuai instruksi, **tidak** diimplementasikan: kontak, geometri jersey, jahitan (seams), UI, CAD, termal, CFD, optimasi.

## 8. File hasil

- `results/experiment2-vgpu-hardware.json`: ringkasan semua kupon.
- `coupon/<jalur>-n<N>/`: `problem.json`, `reference.json`, `gpu-result.json`, `result.json` (bin di-gitignore).

## 9. Ringkasan satu paragraf

Newton f32 di GPU (kernel membran WGSL + solver PCG E1 yang tidak diubah) bekerja: setiap kernel cocok dengan f64 sampai beberapa ε, Newton f32 mengikuti
Newton f64 secara kuadratik iterasi demi iterasi, dan jawaban fisiknya hanya berbeda ~10⁻⁶–10⁻⁵ relatif. Target residual 10⁻⁶ hanya tercapai di patch
test. Di kupon tarik, Newton berhenti tepat di lantai yang ditentukan oleh presisi f32 dari koordinat posisi absolut. Tidak ada solver f32 yang bisa
melewati lantai itu tanpa mengubah cara posisi disimpan.
