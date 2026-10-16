# Step 3.13: `gpu/src/newton.ts`, Inti Newton f32 yang Generik

> Kembaran TypeScript dari refactor `newton.rs` (dokumen 3.3). Loop Newton f32 dari Step 2 dipisah menjadi **inti generik** yang tidak tahu
> soal kupon atau drape, ditambah jalur kupon yang perilakunya identik dengan sebelumnya.

---

## 1. Dua antarmuka baru

```ts
/** The GPU surface Newton needs: an evaluate pass filling the solver's vals/rhs, and an energy-change pass. */
export interface NewtonGpu<A, K> {
  readonly backend: Backend<A, K>;
  readonly solver: PreparedSolver<A, K>;
  readonly positions: A;
  readonly direction: A;
  readonly step: A;
  readonly elemDelta: A;
  readonly evaluate: readonly Dispatch<K>[];
  readonly energyChange: readonly Dispatch<K>[];
}

/** Settings and the row → vertex map of one discretised energy. */
export interface NewtonModel {
  readonly newton: NewtonSettings;
  readonly pcg: PcgSettings;
  readonly rowVertex: Uint32Array<ArrayBuffer>;
  readonly vertices: number;
}
```

- `NewtonGpu`: apa yang dibutuhkan Newton dari GPU, yaitu buffer posisi/arah/langkah/delta, dispatch "evaluate" (yang mengisi `vals`/`rhs` solver), dan dispatch
  "energy change". `CouponGpu` (E2) dan `DrapeGpu` (E3) sama-sama memenuhinya. TypeScript memakai **structural typing**, jadi tidak perlu
  `implements`: objek yang punya field yang cocok otomatis kompatibel.
- `NewtonModel`: pengaturan + peta baris→titik. Dibuat dari `CouponProblem` (`couponModel`) atau `DrapeProblem` (`drapeModel`).

## 2. Konteks

```ts
/** Iterates are kept only when an oracle will evaluate every one of them. */
export interface NewtonContext<A, K> {
  readonly gpu: NewtonGpu<A, K>;
  readonly model: NewtonModel;
  readonly records: IterationRecord[];
  readonly iterates: Float32Array<ArrayBuffer>[] | null;
}
```

`iterates: ... | null`: E2 menyimpan semua iterate (puluhan MB). Untuk lintasan drape n65 (240 langkah × sampai 31 iterate × 50 KB) itu ~370 MB, dan n129 lebih dari 1 GB.
Jadi lintasan E3 memakai `null` (tidak disimpan), sedangkan probe E3 (5 langkah) menyimpannya.

```ts
function record<A, K>(context: NewtonContext<A, K>, entry: IterationRecord, positions: Float32Array<ArrayBuffer>): void {
  context.records.push(entry);
  context.iterates?.push(Float32Array.from(positions));
}
```

`?.` (*optional chaining*): kalau `iterates` null, tidak terjadi apa-apa.

## 3. Yang di-export

| Fungsi | Dipakai oleh |
|---|---|
| `solveStep(context, positions, scale, step)` | jalur kupon (E2) dan langkah drape (E3) |
| `assemble(gpu, positions)` | skala residual ‖∇Φ(x_n)‖ di E3 |
| `norm32` | norma f32 |

Isi `solveStep`, `update`, `lineSearch`, `energyChange` **tidak berubah logikanya**. Hanya `context.problem.manifest.newton` diganti `context.model.newton`, dan
`scatter`/`advance` membaca `model.rowVertex`.

## 4. Jalur kupon tetap sama

```ts
const couponModel = (problem: CouponProblem): NewtonModel =>
  ({ newton: problem.manifest.newton, pcg: problem.manifest.pcg, rowVertex: problem.u32.row_vertex, vertices: problem.manifest.vertices });

export async function solvePath<A, K>(gpu: CouponGpu<A, K>, problem: CouponProblem): Promise<PathResult> {
  const context: NewtonContext<A, K> = { gpu, model: couponModel(problem), records: [], iterates: [] };
  ...
}
```

`stepScale` sekarang menerima `gpu` dan `problem` terpisah (dulu dari context). Regresi E2: `uniaxial-n33` dijalankan ulang, dan hasilnya identik dengan commit E2
kecuali waktu.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `NewtonGpu` | antarmuka struktural; CouponGpu dan DrapeGpu otomatis cocok |
| `NewtonModel` | pengaturan + peta baris |
| `iterates` opsional | lintasan panjang tidak menyimpan iterate |
| Logika tetap | regresi E2 identik |
