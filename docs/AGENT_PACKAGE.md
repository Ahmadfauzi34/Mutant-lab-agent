# Agent Package

The portable result of training should not contain the whole laboratory.

## Thin package

Use this when a compatible runtime is already installed:

```text
my-agent/
├── brain.db
└── AGENT_MANIFEST.json
```

## Self-contained package

Use this when the target environment does not already provide the runtime:

```text
my-agent/
├── runtime/
├── brain.db
└── AGENT_MANIFEST.json
```

Training datasets, incubator logs, failed mutants, sandbox internals, and development checkpoint history remain in the lab/development environment.

A manifest should at minimum pin:

- agent identity;
- compatible runtime semantic version;
- brain schema/version;
- seed lineage;
- validation profile;
- SHA-256 of the brain artifact.
