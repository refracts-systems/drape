# Step 3.19: `run-e3.sh` dan Hasil Experiment 3

> Dokumen penutup Step 3: cara menjalankan, hasil terukur per gate, dan artinya untuk program jersey.

---

## 1. Skrip

```bash
ADAPTER="${1:-hardware}"
SIZES="${SIZES:-33 65}"
GPU_ONLY="${GPU_ONLY:-129}"
...
for n in $SIZES $GPU_ONLY; do
  dir="drape/drape-n$n"
  reference=$([[ " $GPU_ONLY " == *" $n "* ]] && echo false || echo true)
  [ -f "$dir/problem.bin" ] || "$ORACLE" e3-generate --n "$n" --reference "$reference" --out "$dir"
  (cd gpu && node src/node/run-e3.ts --case "../$dir" --adapter "$ADAPTER")
  "$ORACLE" e3-compare --dir "$dir" > /dev/null
done
"$ORACLE" e3-summarize --dir drape --out "results/experiment3-vgpu-$ADAPTER.json"
```

Struktur sama dengan `run-e1.sh`/`run-e2.sh`. Yang baru adalah `GPU_ONLY`: ukuran yang tidak mendapat lintasan f64. `[[ " $GPU_ONLY " == *" $n "* ]]` adalah trik bash untuk
"apakah n ada di daftar"; spasi di kedua sisi mencegah 29 cocok dengan 129.

## 2. Hardware

RTX 3050 Laptop GPU, driver 32.0.16.1714, D3D12 lewat vgpu. Ditolak: iGPU AMD dan software renderer. Run final dijalankan tanpa proses berat lain di CPU.

## 3. Hasil per gate

| Gate | n33 | n65 | n129 |
|---|---|---|---|
| FD gradien Φ ≤ 10⁻⁶ | 6,0×10⁻⁹ ✓ | 3,2×10⁻⁹ ✓ | ✓ |
| FD Hessian Φ (eksak) ≤ 10⁻⁵ | 2,7×10⁻¹⁰ ✓ | 5,7×10⁻¹⁰ ✓ | ✓ |
| Identitas energi hinge ≤ 10⁻¹² | 2,0×10⁻¹⁶ ✓ | 1,6×10⁻¹⁶ ✓ | ✓ |
| Kernel WGSL vs f64 ≤ 10⁻⁵ | 2,7×10⁻⁷ ✓ | 2,6×10⁻⁷ ✓ | 2,5×10⁻⁷ ✓ |
| f64 konvergen atau di lantai | **197/240 ✗** | **170/240 ✗** | – |
| Konsistensi satu langkah ≤ 10⁻³ | 1,8×10⁻⁶ sampai 4,7×10⁻⁵ ✓ | 1,7×10⁻⁶ sampai 4,6×10⁻⁵ ✓ | – |
| f32 Newton: 10⁻⁶ atau lantai ≤ 2× representasi | ✓ (lantai = representasi) | ✓ | – |
| Energi f64 tidak naik | ✓ (selalu turun) | ✓ | – |
| Energi f32 tidak naik (≤ 10⁻⁶) | ✓ (selalu turun) | ✓ | ✓ |
| Pertumbuhan PCG < 3× | 2,14× (n33→n65) ✓ | 1,73× (n65→n129) ✓ | |
| ms per langkah (anggaran 4,17 ms) | 903 | 1742 | 7180 |

PCG per solve Newton: 216 (n33), 463 (n65), 799 (n129). Residual sejati PCG terburuk: 5×10⁻⁵, 1,4×10⁻⁴, 7,1×10⁻⁴, semuanya jauh di bawah forcing term 10⁻².

## 4. Lantai f32 vs lantai representasi (probe)

| Probe | n33: lantai f32 / representasi | n65: lantai f32 / representasi |
|---|---|---|
| 1 | 2,71×10⁻² / 2,73×10⁻² | 8,47×10⁻² / 8,48×10⁻² |
| 10 | 5,29×10⁻³ / 5,27×10⁻³ | 2,14×10⁻² / 2,14×10⁻² |
| 60 | 1,09×10⁻³ / 1,08×10⁻³ | 4,37×10⁻³ / 4,42×10⁻³ |
| 120 | 1,84×10⁻³ / 1,90×10⁻³ | 7,43×10⁻³ / 7,46×10⁻³ |
| 240 | 1,26×10⁻³ / 1,28×10⁻³ | 5,18×10⁻³ / 5,17×10⁻³ |

Sama seperti E2: Newton f32 berhenti **tepat** di residual terbaik yang bisa dicapai posisi f32. Lantai relatifnya jauh lebih besar daripada E2 (hingga 8×10⁻²) karena
skala residual ‖∇Φ(x_n)‖ kecil dibanding suku inersia m/h²·x, yang besar karena h² kecil. Walau begitu, galat **posisi** satu langkah hanya ~10⁻⁶–10⁻⁵ dari
perpindahan langkah itu. Residual relatif itu sangat ketat dibanding galat fisiknya.

Contoh probe 10 (n33), residual per iterasi:

```
f32 GPU           : 2,5×10⁻¹ → 1,2×10⁻² → 5,3×10⁻³ (lantai)
f64 dari iterate  : 2,5×10⁻¹ → 1,2×10⁻² → 5,3×10⁻³      ← identik: residual f32 jujur
f64 referensi     : 2,5×10⁻¹ → 1,1×10⁻² → 1,8×10⁻⁴ → 1,5×10⁻⁷ → 3,4×10⁻⁹
```

## 5. Trigger berbasis bukti

| Trigger | Terpicu? | Bukti | Efek terukur |
|---|---|---|---|
| Tangen tidak PD → Hessian terjepit + Gauss–Newton | **ya** | n33 langkah 87: `pivot -1.21e-1` | f64 rata-rata 16,2 (n33) / 18,7 (n65) iterasi per langkah; 43 / 70 langkah tidak mencapai 10⁻¹⁰ (terburuk 1,1×10⁻² / 2,2×10⁻⁴); GPU memakai 30 iterasi penuh setiap langkah |
| Konsistensi gagal dan lantai = representasi → simpan perpindahan | **tidak** | konsistensi lulus | – |
| Pertumbuhan iterasi ≥ 3× → multigrid | **tidak** | 2,14× dan 1,73× | – |

## 6. Lintasan penuh (diagnostik)

- Bentuk akhir kain f32 vs f64 di t = 1 s: maksimum 8,2×10⁻⁴ (n33) dan 2,9×10⁻³ (n65) dari ukuran lembar, kira-kira 0,4 mm dan 1,5 mm. Kecil, walau lintasannya
  berkerut dan chaos.
- Energi total f32 dan f64 turun di setiap langkah: tidak pernah naik, tidak ada ledakan.

## 7. Apa artinya

**Terbukti**: GPU f32 (WebGPU via vgpu) cukup **akurat** dan **stabil** untuk kain dinamis 3D yang menekuk dan berkerut. Setiap kernel cocok dengan f64 sampai ~3×10⁻⁷,
setiap langkah GPU dari keadaan yang sama hampir identik dengan f64, dan energi selalu turun.

**Belum terpecahkan**:

1. **Konvergensi Newton di fase kompresi.** Tangen proyeksi (efek trigger) membuat Newton linear; f64 pun tidak konvergen di 18–29% langkah dalam 30 iterasi.
2. **Kecepatan.** 903 sampai 7180 ms per langkah, atau 200–1700× di atas anggaran real-time 4,17 ms. Penyebabnya 30 iterasi Newton × 200–800 iterasi PCG per langkah.
   Pertanyaan kecepatan E1 tetap terbuka.

**Perubahan arsitektur yang dipaksa bukti**: hanya satu, yaitu tangen proyeksi (trigger pertama). Perbaikan `atan2` adalah perbaikan bug presisi, bukan perubahan arsitektur.

## 8. File hasil

- `results/experiment3-vgpu-hardware.json`: ringkasan + pertumbuhan PCG.
- `drape/drape-n<N>/`: `problem.json`, `reference.json` (n33, n65), `gpu-result.json`, `result.json`. File `.bin` di-gitignore.
