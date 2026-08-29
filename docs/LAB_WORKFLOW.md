# Lab Workflow

The default workflow is intentionally conservative:

```text
promoted seed
  ↓
spawn isolated mutant
  ↓
train with controlled data / sandbox
  ↓
evaluate against required boundaries
  ↓
freeze portable brain state
  ↓
export agent package
```

The canonical runtime is not rewritten merely because a mutant learns.

Architecture changes follow a separate path: isolate the change, run inherited regression and relevant sandboxes, produce a full checkpoint ZIP, and only then promote a clean result back to this repository.

GitHub updates should be infrequent milestone promotions, not an experiment log.
