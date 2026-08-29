# Concepts

## Runtime / seed

The canonical runtime is the shared cognitive architecture. It is promoted only after validation and should remain clean during ordinary agent training.

## Brain

A brain is the portable cognitive state, normally stored as a `.db` portable-state artifact. Different brains may share the same compatible runtime.

## Mutant

A mutant is an isolated experimental descendant of a known seed. Ordinary training should primarily change its cognitive state. Architecture mutation is exceptional and must not silently modify the canonical runtime.

## Incubator

The incubator is temporary working space for a mutant. It may contain local state, generated data, logs, and experimental code. These are ignored by Git by default.

## Sandbox

A sandbox is a controlled environment used for training, boundary checking, or evaluation. Standard sandboxes protect architectural invariants; experimental sandboxes may be added without changing the seed.

## Checkpoint vs repository

- **Checkpoint ZIP:** complete continuation/recovery artifact for developers and future AI sessions.
- **GitHub repository:** clean promoted runtime/lab product.
- **Agent package:** minimal portable result, usually a brain database plus manifest and optionally a compatible runtime.

## Multi-agent

Multi-agent operation is optional. The architecture should allow multiple isolated brains to use the same runtime, but no orchestration layer is required by the base repository.
