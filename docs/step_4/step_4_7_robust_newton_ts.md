# Step 4.7: `gpu/src/robust-newton.ts`, Newton f32 yang Robust dan Hemat

> Inti Step 4 di GPU. Newton yang sama dengan E2/E3 (rakit → PCG → Armijo → maju), ditambah tiga saklar yang bisa dinyalakan satu per satu untuk ablation:
> **aturan berhenti berbasis update**, **forcing Eisenstat–Walker**, dan **regularisasi μ**.

---

## 1. Aturan dari `problem.json`

```ts
const RULE_KEYS = ["eta_start", "eta_min", "eta_max", "eta_gamma", "eta_safeguard", "update_stop", "mu_start", "mu_cap", "kernel_mu"] as const;
export type RobustRules = Readonly<Record<(typeof RULE_KEYS)[number], number>>;

export function parseRobustRules(text: string): RobustRules { ... numbers(Reflect.get(value, "robust"), RULE_KEYS, "robust") ... }
```

Konstanta aturan dibaca dari bagian `"robust"` yang ditulis oracle (`e4_rules::settings_json`). GPU tidak punya angka ajaib sendiri.

## 2. Saklar ablation

```ts
/** Ablation switches: floor-aware stop, Eisenstat–Walker forcing, K + μ·M/h² regularisation. */
export interface RobustSettings {
  readonly floorStop: boolean;
  readonly forcing: boolean;
  readonly regularize: boolean;
}
```

| Rung | floorStop | forcing | regularize | tangen |
|---|---|---|---|---|
| A | – | – | – | proyeksi (E3) |
| B | ✓ | – | – | proyeksi |
| C | ✓ | ✓ | – | proyeksi |
| D | ✓ | ✓ | ✓ | eksak |

## 3. Catatan tambahan per iterate

```ts
/** What each iterate's decision rested on, recorded next to its IterationRecord for the oracle to replay. */
export interface RobustRecord {
  readonly eta: number;
  readonly mu: number;
  readonly updateInf: number;
  readonly stepInf: number;
  readonly retries: number;
}
```

Setiap `IterationRecord` (format E2/E3 yang sama) mendapat pasangan `RobustRecord`. Dengan **input keputusan** tercatat (η, μ, ‖αd‖∞, ‖x − x_n‖∞), oracle bisa memeriksa bahwa
GPU menerapkan aturan dengan benar, bukan sekadar percaya.

Analogi backend: **audit log** yang mencatat bukan hanya "apa yang terjadi" tetapi "berdasarkan data apa keputusan diambil".

## 4. Forcing dalam f32

```ts
/** Eisenstat–Walker choice 2 in f32: η = γ·ratio², safeguarded by γ·η_prev² above the threshold, clamped. */
export function nextForcing(rules: RobustRules, previous: number, ratio: number): number {
  const eta = f32(rules.eta_gamma * f32(ratio * ratio));
  const safeguard = f32(rules.eta_gamma * f32(previous * previous));
  const guarded = safeguard > rules.eta_safeguard ? Math.max(eta, safeguard) : eta;
  return Math.min(rules.eta_max, Math.max(rules.eta_min, guarded));
}
```

Kembaran `e4_rules::next_forcing` dengan `Math.fround` per operasi (`f32 = Math.fround`).

## 5. Mencari arah dengan regularisasi

```ts
async function solveDirection<A, K>(context, positions, rhs, state): Promise<{ readonly run: SolveRun; readonly retries: number } | null> {
  const settings = { tolerance: state.eta, maxIterations: ..., chunkIterations: CHUNK_ITERATIONS };
  for (let retries = 0; ; retries += 1) {
    const run = await inPhase(context.clock, "pcg", () => solveOnce(context.gpu.solver, settings));
    const failed = run.status === "breakdown" || !(-dot32(rhs, run.solution) < 0);
    if (!failed || !context.settings.regularize) return { run, retries };
    if (state.mu >= context.rules.mu_cap) return null;
    state.mu = raiseMu(context.rules, state.mu);
    writeMu(context, state.mu);
    await inPhase(context.clock, "assembly", () => assemble(context.gpu, positions));
  }
}
```

- **Toleransi PCG = η** (forcing) atau 10⁻⁷ (tanpa forcing). Solver E1 menerimanya lewat `SolveSettings` tanpa diubah.
- **Kegagalan** = PCG melaporkan breakdown (pᵀAp ≤ 0, tanda matriks tidak PD; dokumen 1.10) atau arah tidak turun.
- Dengan regularisasi: μ naik, ditulis ke buffer, **matriks dirakit ulang** (hanya blok inersia yang berubah), lalu dicoba lagi.
- Tanpa regularisasi (rung A–C): dikembalikan apa adanya, dan line search yang akan gagal, persis seperti E3.

Detektor di GPU adalah breakdown PCG, bukan Cholesky. Cholesky tidak ada di GPU.

## 6. Satu langkah waktu

```ts
export async function solveStepRobust<A, K>(context, positions, scale, step, origin): Promise<void> {
  const state: StepState = { mu: 0, eta: context.rules.eta_start, previousNorm: Number.NaN };
  for (let iteration = 0; ; iteration += 1) {
    writeMu(context, state.mu);
    const rhs = await inPhase(context.clock, "assembly", () => assemble(context.gpu, positions));
    const norm = norm32(rhs);
    const relative = f32(norm / scale);
    ...
    if (relative <= tolerance || iteration === max) return finish(...converged / iteration_limit...);
    state.eta = forcingFor(context, state, norm, iteration);
    state.previousNorm = norm;
    const updateInf = await update(context, positions, rhs, state, where);
    if (updateInf === null) return;
    if (context.settings.floorStop && updateInf <= context.rules.update_stop * stepInf(context, positions, origin)) {
      return finishByUpdate(context, { step, iteration: iteration + 1 }, scale, state, positions);
    }
  }
}
```

1. Tulis μ, lalu rakit dan ukur residual.
2. Berhenti kalau residual ≤ 10⁻⁶ atau batas 30 iterasi.
3. Hitung η.
4. Update: arah (dengan regularisasi), line search, catat, maju, lalu relax μ setelah langkah penuh.
5. **Aturan update**: kalau ‖αd‖∞ ≤ 10⁻⁴·‖x − x_n‖∞, rakit sekali lagi untuk mencatat residual akhir, lalu berhenti dengan status `update_rule`.

`origin` adalah x_n (keadaan awal langkah waktu), dibutuhkan untuk ‖x − x_n‖∞.

`StepState` adalah objek kecil yang **dimutasi** (μ, η, norma sebelumnya). Ini sengaja: tiga nilai yang berubah bersama di sepanjang iterasi lebih jelas sebagai satu
objek daripada tiga `let` yang dioper ke mana-mana.

## 7. Normal ∞ dan langkah

```ts
function infNorm(values: Float32Array): number { ... max |v| ... }

/** ‖x − x_n‖∞ over the unknowns. */
function stepInf(context, positions, origin): number { ... max atas row_vertex dan 3 komponen ... }
```

Norma maksimum (∞) dipilih, bukan norma 2, supaya aturan berarti "**tidak ada satu titik pun** yang masih bergerak lebih dari 10⁻⁴ dari langkahnya", bukan
rata-rata.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Tiga saklar | ablation bertahap A → D |
| η sebagai toleransi PCG | solver E1 tidak diubah |
| Breakdown sebagai detektor | μ naik, rakit ulang, coba lagi |
| Aturan update | berhenti dalam satuan meter |
| Audit log | input setiap keputusan dicatat untuk replay oracle |
