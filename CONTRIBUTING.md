# Contributing

**External contributions are not accepted yet.** Pull requests from people outside the organization will be closed until a
Contributor License Agreement (CLA) is in place. The CLA is needed so that the project can keep offering the engine under
Apache-2.0 and license the editor and hosted service separately without legal ambiguity.

Issues (bug reports, questions, measurements on other hardware) are welcome.

## Working rules for the engine

- The f64 oracle is the reference. Never loosen a tolerance to obtain a pass; report the failing gate instead.
- Do not change the behaviour of earlier experiments (E1–E6 outputs must stay bit-identical except timing).
- Do not run heavy CPU work (oracle generation or scoring) while timing GPU runs.
- Run data (`*.bin`) is not committed; stored oracle references of `drape4-final` and `drape5` are not regenerated casually.
- Before a commit: `cargo test --release` in `engine/oracle` and `npm run typecheck` in `engine/gpu`.
