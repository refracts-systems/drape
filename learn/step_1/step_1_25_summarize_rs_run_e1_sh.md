# Step 1.25: `oracle/src/summarize.rs` dan `run-e1.sh`, Ringkasan dan Skrip Ujung ke Ujung

> Gate pertumbuhan iterasi tidak bisa dinilai per kasus; ia butuh beberapa ukuran mesh sekaligus. `summarize.rs` mengumpulkan
> semua `result.json` dan menilainya. `run-e1.sh` merangkai seluruh pipeline.

---

## 1. `summarize.rs`

### Mengumpulkan

```rust
fn load_rows(dir: &Path) -> Result<Vec<Row>, String> {
    let mut rows: Vec<Row> = fs::read_dir(dir)?
        .filter_map(|entry| fs::read_to_string(entry.ok()?.path().join("result.json")).ok())
        .map(|text| Row { case: string(&text, "", "case"), columns: number(&text, "", "grid_columns"), text })
        .collect();
    rows.sort_by(|a, b| a.case.cmp(&b.case).then(a.columns.total_cmp(&b.columns)));
    Ok(rows)
}
```

Semua subfolder `cases/*/result.json` dibaca, lalu diurutkan berdasarkan kasus, kemudian ukuran grid.

### Pertumbuhan iterasi

```rust
fn scaling(rows: &[Row]) -> Vec<Json> {
    rows.windows(2)
        .filter(|pair| pair[0].case == pair[1].case)
        .map(|pair| {
            let growth = iterations(pair[1]) / iterations(pair[0]);
            ... .flag("within_limit", growth <= ITERATION_GROWTH_LIMIT)
        })
        .collect()
}
```

`windows(2)` menghasilkan pasangan bertetangga (n33→n65, n65→n129, n129→n257). Untuk kasus yang sama, hitung rasio iterasi.

**Kenapa ini gate?** Setiap kali mesh diperhalus 2×, κ naik ~4× (operator orde 2), sehingga iterasi CG naik ~2× (√4). Kalau naik ≥ 3×,
solver tidak akan skala ke resolusi lebih tinggi, dan itu pemicu untuk menambah **multigrid**. Hasil:

| Kasus | 33→65 | 65→129 | 129→257 |
|---|---|---|---|
| tension | 2,01 | 2,08 | 2,45 |
| dynamic | 2,54 | 2,12 | 2,15 |
| slack (diagnostik) | 3,99 | 4,13 | (batas iterasi) |

Tension dan dynamic < 3, jadi multigrid **tidak** dipicu. Slack ~4, sesuai sifat orde-4 tekukan, dan memang kasus gagal yang diketahui.

### Tabel cetak

```
tension  n=257  N=195075  it=1545  res=7.84e-3  err64=8.45e-4  kappa=4.70e4  ms/it=7.93e-1  speedup=6.57e0  conv/forcing/f32it/faithful/det/speed=Y/Y/Y/Y/Y/n
```

Satu baris per kasus, dengan semua flag gate sebagai Y/n. Dari tabel inilah keputusan dibaca.

### Output

`results/experiment1-<backend>-<adapter>.json` berisi `scaling` dan semua `result.json` sebagai satu file.

## 2. `run-e1.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${1:-webgpu}"
ADAPTER="${2:-hardware}"
CASES="${CASES:-tension dynamic slack}"
SIZES="${SIZES:-33 65 129}"
REPEATS="${REPEATS:-3}"
REPLACE_EVERY="${REPLACE_EVERY:-0}"
REPLACE_ABOVE="${REPLACE_ABOVE:-0}"
ORACLE=./oracle/target/release/oracle
RUNNER=$([ "$BACKEND" = vgpu ] && echo src/node/run-vgpu.ts || echo src/node/run-browser.ts)

cargo build --release --manifest-path oracle/Cargo.toml
for kind in $CASES; do
  for n in $SIZES; do
    dir="cases/$kind-n$n"
    [ -f "$dir/reference.bin" ] || "$ORACLE" generate --case "$kind" --n "$n" --out "$dir" --replace-every "$REPLACE_EVERY" --replace-above "$REPLACE_ABOVE"
    (cd gpu && node "$RUNNER" --case "../$dir" --adapter "$ADAPTER" --repeats "$REPEATS")
    "$ORACLE" compare --dir "$dir" > /dev/null
  done
done
"$ORACLE" summarize --dir cases --out "results/experiment1-$BACKEND-$ADAPTER.json"
```

Baris demi baris:

- `set -euo pipefail`: berhenti di error pertama (`-e`), variabel tak terdefinisi adalah error (`-u`), error di pipe tidak disembunyikan
  (`pipefail`). Mode "strict" bash.
- `cd "$(dirname "$0")"`: pindah ke folder skrip, supaya bisa dipanggil dari mana pun.
- `${VAR:-default}`: env var dengan default. Konfigurasi lewat environment, seperti aplikasi 12-factor.
- `[ -f reference.bin ] || generate`: **cache**. Oracle n257 butuh menit; kalau referensi sudah ada, lewati. Konsekuensi penting: kalau
  pengaturan replacement diubah, referensi lama harus dihapus dulu (sesi kerja melakukan itu dengan menyalin baseline ke `cases-baseline/`).
- `(cd gpu && node ...)`: subshell supaya `cd` tidak memengaruhi baris berikutnya.
- `REPLACE_EVERY` dan `REPLACE_ABOVE` ditambahkan di sesi ini dan diteruskan ke `generate`; GPU membacanya dari `problem.json`.

Di Windows dijalankan lewat Git Bash.

### Contoh pemakaian di sesi kerja

```bash
# baseline (tanpa replacement)
CASES="tension dynamic slack" SIZES="33 65 129 257" ./run-e1.sh webgpu hardware
# dengan replacement
CASES="tension dynamic slack" SIZES="33 65 129 257" REPLACE_EVERY=50 REPLACE_ABOVE=1e-2 ./run-e1.sh webgpu hardware
CASES="tension dynamic slack" SIZES="33 65 129 257" REPLACE_EVERY=50 REPLACE_ABOVE=1e-2 ./run-e1.sh vgpu hardware
```

Catatan operasional: generate oracle n257 sengaja tidak dijalankan bersamaan dengan pengukuran GPU, karena beban CPU dari Cholesky bisa
mengganggu waktu submit GPU dan waktu CPU patokan.

## 3. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `windows(2)` | pasangan ukuran bertetangga per kasus |
| Pertumbuhan < 3× | solver skala; multigrid tidak perlu |
| Tabel Y/n | satu baris per kasus, keputusan dibaca dari sini |
| Strict bash | berhenti di error pertama |
| Cache referensi | lewati generate kalau sudah ada; hapus kalau pengaturan berubah |
