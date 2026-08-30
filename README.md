# Mutant Lab Agent

Mutant Lab Agent is a clean laboratory/runtime repository for building and training portable cognitive agents from one shared architecture.

The repository intentionally keeps the product surface small:

- `agen_lab/` — canonical cognitive runtime.
- `core/` — compatibility shim required by the runtime.
- `lab/` — lightweight incubation, sandbox, and training workspace contracts.
- `brains/` — exported/seed cognitive-state placement convention; trained `.db` files are not committed by default.
- `tests/` — small product smoke tests.
- `docs/` — user-facing concepts and packaging guidance.

Development checkpoints, long audit trails, generated datasets, active mutants, and training logs stay outside the product repository. A checkpoint ZIP is the development-continuity artifact; this repository is the clean promoted result.

## Current seed runtime

- Semantic runtime: **V2.42**
- Lineage: Canonical R1
- Promotion validation before this repository seed: **1201 / 1201 PASS**

## Core idea

```text
clean shared runtime
       +
portable brain.db
       =
usable agent
```

Training normally mutates cognitive state, not the canonical runtime. Experimental architecture changes belong in an isolated incubator and must pass the standard sandbox/regression boundary before promotion.

## Quick check

```bash
python -m unittest tests.test_runtime_smoke -v
```

## Lab workflow

```text
Seed runtime
   ↓ spawn
Incubator mutant
   ↓ train in controlled data/sandboxes
Evaluate
   ↓
Freeze brain.db
   ↓
Export agent capsule
```

Multi-agent use is optional. Multiple agents can share this runtime while loading different cognitive-state databases.

See `docs/CONCEPTS.md` and `docs/AGENT_PACKAGE.md` for the minimal contracts.

## Canonical baseline sandbox

The root `sandboxes/` tree and `run_sandboxes.py` / `run_game_e2e.py` are promoted verbatim from the Seed G0 V2.42 checkpoint. They are a protected behavioral baseline for the canonical architecture, not training data.

```bash
python run_sandboxes.py
python run_game_e2e.py
```

Experimental or domain training sandboxes belong under `lab/sandboxes/`; they do not replace the canonical baseline. A candidate that breaks the root baseline is not promotion-safe even if it performs well on its training curriculum.

## Minimal local lab

`mutant_lab.py` keeps the active experiment local and small. It verifies the frozen baseline fingerprint before spawning a mutant, creates a portable `brain.db` under the ignored `lab/incubator/` workspace, and can run the canonical baseline inside a temporary directory so sandbox-generated artifacts do not dirty the repository.

```bash
python mutant_lab.py baseline
python mutant_lab.py spawn python-specialist --purpose "learn Python domain experience"
python mutant_lab.py inspect python-specialist
```

The default 16×16 reference space is only a convenient reference machine for a fresh brain and can be changed with `--width` / `--height`. Architecture mutation is deliberately a separate runtime experiment; this small tool spawns state or knowledge mutants against the frozen Seed G0 runtime.
