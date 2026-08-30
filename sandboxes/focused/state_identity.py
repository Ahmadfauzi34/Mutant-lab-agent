import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import agen_lab as core


def build(name):
    path = Path(tempfile.gettempdir()) / f'{name}_v228_state_user.sqlite3'
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return core.IntegratedCognitiveAgent(
        name,
        4,
        4,
        epistemic_archive_path=str(path),
    )


STATE_A = 'branch=main|dirty=0|tests=green'
STATE_B = 'tests=green|dirty=0|branch=main'
STATE_DIFFERENT = 'branch=main|dirty=1|tests=green'
GOOD = 'A_GOOD'
BAD = 'Z_BAD'


def train(agent, state):
    # Force actual experience for both actions without relying on exploration.
    for _ in range(8):
        d = agent.choose_action(state, [GOOD])
        agent.record_decision_outcome(d.decision_id, 1.0)
        agent.record_world_model_outcome(state, GOOD, 1.0, True)

        d = agent.choose_action(state, [BAD])
        agent.record_decision_outcome(d.decision_id, 0.0)
        agent.record_world_model_outcome(state, BAD, 0.0, False)


def snapshot(agent, state):
    decision = agent.choose_action(state, [GOOD, BAD])
    good_prediction = agent.predict_outcome(state, GOOD)
    bad_prediction = agent.predict_outcome(state, BAD)
    return {
        'state': state,
        'state_key': decision.state_key,
        'selected': decision.selected_action,
        'utilities': decision.utility_estimates,
        'good_samples': good_prediction.sample_count,
        'bad_samples': bad_prediction.sample_count,
    }


baseline = build('state_fragmentation_baseline')
canonical = build('state_fragmentation_canonical')
canonical.register_state_equivalence(
    canonical_id='repo:clean-main-green',
    equivalence_fingerprint='repo-state-v1:branch+dirty+tests',
    aliases=(STATE_A, STATE_B),
    note='order-only representation difference',
)

train(baseline, STATE_A)
train(canonical, STATE_A)

baseline_equivalent = snapshot(baseline, STATE_B)
canonical_equivalent = snapshot(canonical, STATE_B)
canonical_different = snapshot(canonical, STATE_DIFFERENT)

results = {
    'baseline_equivalent_state_fragmented': (
        baseline_equivalent['good_samples'] == 0
        and baseline_equivalent['bad_samples'] == 0
        and baseline_equivalent['selected'] == BAD
    ),
    'canonical_equivalent_state_reuses_q': (
        canonical_equivalent['utilities'][GOOD] > 0.9
        and canonical_equivalent['utilities'][BAD] < 0.1
        and canonical_equivalent['selected'] == GOOD
    ),
    'canonical_equivalent_state_reuses_world_model': (
        canonical_equivalent['good_samples'] == 8
        and canonical_equivalent['bad_samples'] == 8
    ),
    'raw_context_preserved_with_canonical_learning_key': (
        canonical_equivalent['state'] == STATE_B
        and canonical_equivalent['state_key'] == 'repo:clean-main-green'
    ),
    'different_semantic_state_not_merged': (
        canonical_different['state_key'] == STATE_DIFFERENT
        and canonical_different['good_samples'] == 0
        and canonical_different['bad_samples'] == 0
        and canonical_different['selected'] == BAD
    ),
    'no_belief_context_shift_from_state_aliasing': (
        canonical.belief_contexts.current_id == 'ctx-0'
    ),
}

print('=== STATE IDENTITY SANDBOX V2.28 ===')
print('baseline equivalent:', json.dumps(baseline_equivalent, sort_keys=True))
print('canonical equivalent:', json.dumps(canonical_equivalent, sort_keys=True))
print('canonical different:', json.dumps(canonical_different, sort_keys=True))
print('\nRESULTS')
for name, passed in results.items():
    print(('PASS' if passed else 'FAIL'), '|', name)

if not all(results.values()):
    raise SystemExit(1)
