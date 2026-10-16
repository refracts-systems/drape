# Step 4.5: `drape-gpu.ts`, `drape-step.ts`, `newton.ts` (Perubahan Kecil Step 4)

> Perubahan "pembuka pintu": buffer μ, pilihan varian tangen per run, satu daftar dispatch baru, dan fungsi-fungsi primitif yang di-export supaya bisa dirakit
> ulang oleh Newton E4. Tidak ada perubahan logika.

---

## 1. `drape-gpu.ts`

### Buffer μ

```ts
    regularization: backend.array("regularization", new Float32Array(1), "read"),
```

Diikat ke kernel inersia (`regularization: a.regularization`) dan diekspos di `DrapeGpu.regularization`. Dibuat `"read"` karena shader hanya membacanya; host tetap
bisa menulisnya lewat `backend.write` (`queue.writeBuffer`).

### Varian tangen sebagai parameter

```ts
/** `projected` picks the simulation tangent (clamped membrane + Gauss–Newton hinge); it defaults to the problem's own kind. */
export async function prepareDrapeGpu<A, K>(backend, problem, sources, projected = problem.manifest.tangent === "projected"): Promise<DrapeGpu<A, K>>
```

Parameter **default** TypeScript: E3 memanggil tanpa argumen keempat (perilaku lama), sedangkan E4 mengirim `true` (rung A–C) atau `false` (rung D).

### Daftar dispatch inersia

```ts
    inertia: [{ kernel: inertia, workgroups: n.rows }],
```

Untuk cek kernel inersia dengan μ uji (dokumen 4.4).

## 2. `drape-step.ts`

`prediction`, `velocityOf`, `State`, dan `drapeModel` diberi `export`. E4 memakai prediksi x̂ dan kecepatan yang **sama persis** dengan E3, sehingga tidak ada dua
implementasi yang bisa berbeda diam-diam.

## 3. `newton.ts`

```ts
export const STEP_STATUS = { running: 0, converged: 1, iteration_limit: 2, line_search_failed: 3, update_rule: 4, regularization_cap: 5 } as const;
```

Dua status baru:

- `update_rule` (4): berhenti karena koreksi sudah kecil (dokumen 4.2).
- `regularization_cap` (5): μ mencapai batas dan arah masih tidak valid.

Fungsi yang sekarang di-export: `dot32`, `energyChange`, `lineSearch`, `scatter`, `advance`, `record`, `terminal`, `linearStatus`, tipe `LineSearch` dan
`StepStatus`. `solveStep` E2/E3 tidak diubah.

Ini prinsip **open/closed**: modul terbuka untuk dipakai ulang (export), tertutup untuk perubahan (logika lama utuh).

## 4. Ringkasan

| File | Perubahan |
|---|---|
| `drape-gpu.ts` | buffer μ, parameter `projected`, dispatch `inertia` |
| `drape-step.ts` | export helper prediksi/kecepatan |
| `newton.ts` | status 4/5, export primitif |
