# editor

The browser Jersey Editor (two sewn panels on an analytic body first, then real fabric data) is **not developed in this repository**.
Its code will live in a separate **private** repository that consumes the engine as a dependency (package or submodule). This folder
only documents the plan and must stay free of editor code: removing a file from a public repository does not remove it from the history.

## Licensing and product plan (decided 2026-10-18)

- The engine (`engine/`) stays Apache-2.0 and public.
- The editor is a **hosted product**: first a subscription editor with the simulation running in the cloud, then an embeddable
  widget/SDK for web shops (usage- or per-shop pricing).
- The editor's licence is **not decided yet** and is independent of the engine's: open between proprietary and source-available
  (BSL/FSL, opening after a few years).
- The engine is used through a narrow interface so the editor repository depends only on that interface.
