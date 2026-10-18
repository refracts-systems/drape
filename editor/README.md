# editor

Reserved for the browser Jersey Editor (two sewn panels on an analytic body first, then real fabric data). Not started.

## Licensing and product plan (decided 2026-10-18)

- The engine (`engine/`) stays Apache-2.0.
- The editor is built as a **hosted product**: first a subscription editor with the simulation running in the cloud, then an embeddable
  widget/SDK for web shops (usage- or per-shop pricing).
- The editor's licence is **not decided yet** and is independent of the engine's: it is open between a proprietary licence and a
  source-available one (BSL/FSL, opening after a few years). Until it is decided, nothing in this folder is licensed for reuse.
- The editor will use the engine through a narrow interface so that the two can be split into separate packages later.
