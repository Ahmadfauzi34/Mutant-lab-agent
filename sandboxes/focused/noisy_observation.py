import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import agen_lab as core


def build_agent(path):
    agent = core.IntegratedCognitiveAgent(
        'observation_noise_lab',
        4,
        4,
        epistemic_archive_path=path,
    )
    agent.register_source(
        core.SourceProfile(
            'actual_outcome',
            alpha=19,
            beta=1,
        )
    )
    return agent


def signal(
    agent,
    claim,
    polarity,
    t,
    origin,
    quality=1.0,
    note='signal',
):
    audit = agent.consider_context_shift(
        claim_id=claim,
        incoming_polarity=polarity,
        observed_at=t,
        reason=note,
        incoming_strength=1.0,
        source='actual_outcome',
        origin_id=origin,
        observation_quality=quality,
    )
    agent.add_contextual_evidence(
        evidence_id=f'ev-{t}-{origin}',
        source='actual_outcome',
        origin_id=origin,
        claim_id=claim,
        polarity=polarity,
        strength=1.0,
        observed_at=t,
        context_id=audit['current_context'],
        observation_quality=quality,
    )
    return audit


def main():
    archive = str(PROJECT_ROOT / "runtime" / "v2_28_observation_sandbox.sqlite3")
    Path(archive).unlink(missing_ok=True)
    agent = build_agent(archive)
    claim = 'service_requires_token'

    # Strong stable baseline.
    agent.add_contextual_evidence(
        'seed',
        'actual_outcome',
        'seed-run',
        claim,
        +1,
        1.0,
        observed_at=1,
        context_id='ctx-0',
    )
    normal = signal(
        agent, claim, +1, 2, 'normal-2', 1.0,
        'normal confirmation',
    )

    # A low-quality timeout/failure stays in Evidence, but is attenuated and
    # does not even start a context-shift candidate.
    low_quality = signal(
        agent, claim, -1, 3, 'timeout-3', 0.10,
        'partial timeout output',
    )

    # Same technical execution reported twice with contradictory retry output.
    # It is not passed as two context-shift signals because it is one retry
    # family. The Evidence layer quarantines that family and learns that this
    # source is operationally less repeatable.
    agent.add_contextual_evidence(
        'retry-4-positive',
        'actual_outcome',
        'report-a',
        claim,
        +1,
        1.0,
        observed_at=4,
        context_id='ctx-0',
        retry_group_id='retry-run-4',
    )
    agent.add_contextual_evidence(
        'retry-4-negative',
        'actual_outcome',
        'report-b',
        claim,
        -1,
        1.0,
        observed_at=4,
        context_id='ctx-0',
        retry_group_id='retry-run-4',
    )

    old_full = agent.adjudicate_claim(
        claim, 'ctx-0', 4, audit_mode='full'
    )
    old_compact = agent.adjudicate_claim(
        claim, 'ctx-0', 4, audit_mode='compact'
    )

    # Actual persistent regime change: independent high-quality executions.
    change_1 = signal(
        agent, claim, -1, 5, 'change-5', 1.0,
        'persistent change candidate',
    )
    change_2 = signal(
        agent, claim, -1, 6, 'change-6', 1.0,
        'persistent change confirmation',
    )
    new_confirm = signal(
        agent, claim, -1, 7, 'new-7', 1.0,
        'new regime confirmation',
    )

    new_report = agent.adjudicate_claim(
        claim, 'ctx-1', 7, audit_mode='full'
    )
    stability = agent.observation_reliability_state()

    tests = {
        'baseline_confirmation_no_shift':
            normal['shifted'] is False,
        'low_quality_contradiction_ignored_by_shift_detector':
            low_quality['detector_decision']
                == 'weak_contradiction_ignored'
            and low_quality['pending'] is False,
        'retry_conflict_quarantined':
            old_full['retry_quarantined_groups']
                == ['retry-run-4'],
        'old_context_remains_accepted_despite_flaky_retry':
            old_full['evidence_status'] == 'accepted',
        'full_compact_exact_under_noise_handling':
            old_full['support_score']
                == old_compact['support_score']
            and old_full['oppose_score']
                == old_compact['oppose_score'],
        'repeatability_lowered_not_factual_accuracy':
            abs(
                stability['sources']['actual_outcome']
                ['observation_reliability'] - 0.9
            ) < 1e-12
            and abs(
                stability['sources']['actual_outcome']
                ['factual_reliability'] - 0.95
            ) < 1e-12,
        'first_persistent_change_only_pending':
            change_1['pending'] is True
            and change_1['shifted'] is False,
        'second_independent_change_confirms_shift':
            change_2['shifted'] is True
            and change_2['current_context'] == 'ctx-1',
        'new_context_empirical_direction_reversed':
            new_report['evidence_status'] == 'rejected',
        'new_context_baseline_confirmed':
            new_confirm['detector_decision']
                == 'baseline_confirmed',
        'retry_group_status_lives_in_archive':
            stability['retry_group_records'] == 1,
    }

    print('=== NOISY OBSERVATION SANDBOX V2.28 ===')
    print('low quality effective strength:',
          low_quality['effective_signal_strength'])
    print('old context:',
          old_full['evidence_status'],
          old_full['support_score'],
          old_full['oppose_score'])
    print('quarantined retry groups:',
          old_full['retry_quarantined_groups'])
    print('source state:',
          stability['sources']['actual_outcome'])
    print('change1:',
          change_1['detector_decision'],
          change_1['effective_signal_strength'])
    print('change2:',
          change_2['detector_decision'],
          change_2['effective_signal_strength'])
    print('new context:',
          new_report['evidence_status'],
          new_report['support_score'],
          new_report['oppose_score'])

    print('\nRESULTS')
    for name, ok in tests.items():
        print('PASS' if ok else 'FAIL', '|', name)

    result = {
        'tests': tests,
        'old_context': {
            'status': old_full['evidence_status'],
            'support': old_full['support_score'],
            'oppose': old_full['oppose_score'],
            'retry_quarantined_groups':
                old_full['retry_quarantined_groups'],
        },
        'new_context': {
            'status': new_report['evidence_status'],
            'support': new_report['support_score'],
            'oppose': new_report['oppose_score'],
        },
        'source_state':
            stability['sources']['actual_outcome'],
        'current_context':
            agent.belief_contexts.current_id,
    }

    out = PROJECT_ROOT / "runtime" / "noisy_observation_sandbox_v2_28_result.json"
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    if not all(tests.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
