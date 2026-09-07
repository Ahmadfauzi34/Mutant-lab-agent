# Developer Training Baseline

This document is a **developer / evaluator reference** for training agents built on the canonical cognitive runtime.

It is intentionally small. It is **not** a universal curriculum, not an answer key for agents, and not training data to inject into a brain. Domain adapters, tasks, validators, and curricula remain domain-specific.

The purpose of this baseline is to preserve the architectural separations that are easy to lose during handoff or LLM-assisted development.

## What must remain separate

Do not collapse these signals into one reward or one notion of correctness:

- reward / utility != truth;
- reasoning != evidence;
- prediction / simulation != actual outcome;
- action execution success != task progress != final task acceptance;
- reasoning credit != action credit != outcome credit;
- evidence support != formal truth;
- unknown != known-bad;
- failure consequence != failure probability != reward;
- context shift != global deletion;
- historical experience != automatic applicability in a new context.

Only **actual validated environment outcomes** may update empirical action/world-model learning. A prediction, synthetic example, LLM proposal, developer baseline reference, or simulation is not actual experience.

## Minimum training credit discipline

Treat the channels independently.

```text
TASK
  ↓
PROOF STATE
  ↓
HYPOTHESIS / REASONING CLAIM
  ↓
TESTABLE PREDICTION
  ↓
INDEPENDENT VALIDATOR
  ↓
ACTUAL VALIDATION RESULT
  ↓
REASONING CREDIT

ACTION
  ↓
ACTUAL ACTION RESULT
  ↓
ACTION CREDIT

GOAL
  ↓
ACTUAL GOAL ACCEPTANCE
  ↓
OUTCOME CREDIT
```

Recommended interpretation:

- validated reasoning: `+1`;
- falsified reasoning: `-1`;
- unresolved / unvalidated reasoning: `0`;
- successful validated action transition: `+1`;
- validated failed action transition: `-1`;
- not executed / unresolved action: `0`;
- final outcome credit is given only by the task acceptance boundary.

A useful action can coexist with false reasoning. A successful task must not retroactively turn an earlier false claim into a true one.

## Positive and negative training cases

Training/evaluation should contain both:

- positive precedents showing correct architectural use;
- negative counterexamples exposing shortcuts or over-strong claims;
- held-out or shifted contexts when applicability matters.

A negative example constrains a claim; it does not automatically create the opposite global rule.

Green tests are evidence about the tested boundary, not proof of complete correctness.

## Validators and evidence

A validator must be independently discriminating from the executor it evaluates.

Different validator names do not automatically mean independent evidence. Check source/provenance lineage; multiple validators that all derive from the same artifact should not be counted as independent roots.

Evidence acquisition should also be progress-aware. If a source lineage has already been consumed for the same claim/context and its declared reuse policy says another read adds no new information, do not keep proposing that lineage while independent validation routes remain. Preserve its history and learned action utility; new context or genuinely new evidence may make reuse relevant again.

A validated tool/action result is not automatically epistemic evidence. If the result should change a proof state, use an explicit evidence-admission path. Do not infer truth merely because an action succeeded.

Validators are code too. Validate important validators with positive and negative controls; prefer semantic/contract checks over brittle formatting/string matches, and avoid hard-coded internal score thresholds unless that threshold is itself part of a validated public contract.

## LLM-assisted development

An LLM may help with:

- proposing interpretations, hypotheses, tools, probes, examples, or counterexamples;
- producing natural-language output from validated state;
- reading the repository and suggesting candidate integrations.

An LLM must not become, merely by confidence or fluency:

- a truth authority;
- an empirical experience source;
- an automatic task acceptance validator;
- its own response validator.

If a nondeterministic proposer interprets a task, freeze the selected proposal batch for that task attempt so the goal cannot silently move during execution or after restart.

## Runtime / persistence blind spots

These are integration boundaries that can make a correct cognitive core appear to work while the surrounding agent drifts:

- one running agent should have one authoritative writable cognitive database;
- backup/checkpoint/restore copies are recovery or reference artifacts, not competing live brains;
- retries must not silently replay external actions or duplicate empirical learning;
- logical action identity must remain distinct from executable version/fingerprint;
- an accepted task should preserve acceptance provenance, not only an `ACCEPTED` label;
- the original raw task/constraints must remain reconstructible after restart;
- an already validated final response should not be regenerated nondeterministically without new evidence;
- editable workspace files are not automatically the validated runtime source—use validated provenance/hash gates when available.

These are reference warnings, not requirements that every domain use the same harness implementation.

## How to use this baseline

For each domain:

1. define the domain observation/action adapter;
2. define actual validators and acceptance boundaries;
3. train/evaluate the brain using domain experience;
4. inspect proof, evidence, Q/utility, prediction, uncertainty, risk, context, and actual outcomes separately;
5. run positive and negative cases;
6. keep the canonical runtime frozen during ordinary state/knowledge training;
7. treat architecture mutation as a separate, gated experiment.

Do **not** feed this document to the agent as a task-specific cheat sheet when measuring native competence. This baseline tells the developer what to observe and what not to conflate.

## Authority and revision

This baseline is a reference, not an absolute law above the implementation.

If this document conflicts with current validated canonical behavior, stop and investigate whether the cause is stale documentation, regression, semantic change, or misunderstanding. Revise the side proven wrong by validation.

Keep this document small. Add a new item only when its absence is likely to make a future developer or LLM misread or misuse the architecture across domains.
