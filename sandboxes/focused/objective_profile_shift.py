
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import agen_lab as core


INITIAL_PROFILE = core.ObjectiveUtilityProfile(
    profile_id="developer_preferences",
    task_progress_weight=0.10,
    correctness_weight=0.10,
    execution_cost_weight=0.70,
    reversibility_weight=0.05,
    user_acceptance_weight=0.05,
)

FAST_PATCH = {
    "task_progress": 1.00,
    "correctness": 0.20,
    "execution_cost": 0.05,
    "reversibility": 0.50,
    "user_acceptance": 0.30,
}

ROBUST_PATCH = {
    "task_progress": 0.90,
    "correctness": 1.00,
    "execution_cost": 0.60,
    "reversibility": 0.90,
    "user_acceptance": 0.90,
}


def build_agent(
    archive_path: str,
):
    return core.IntegratedCognitiveAgent(
        "developer_preference_shift",
        4,
        4,
        objective_profile=INITIAL_PROFILE,
        epistemic_archive_path=archive_path,
    )


def record_history(
    agent,
    repeats=12,
):
    for _ in range(repeats):
        agent.record_world_model_outcome(
            "repo:bug-open",
            "FAST_PATCH",
            reward=None,
            success=True,
            objective_outcome=FAST_PATCH,
        )
        agent.record_world_model_outcome(
            "repo:bug-open",
            "ROBUST_PATCH",
            reward=None,
            success=True,
            objective_outcome=ROBUST_PATCH,
        )


def main():
    tmp = tempfile.NamedTemporaryFile(
        prefix="v228_pref_",
        suffix=".sqlite3",
        delete=False,
    )
    tmp.close()

    agent = build_agent(
        tmp.name
    )
    record_history(
        agent,
        repeats=12,
    )

    initial_context = (
        agent.belief_contexts.current_id
    )
    initial_choice = (
        agent.choose_action_preference_aware(
            "repo:bug-open",
            [
                "FAST_PATCH",
                "ROBUST_PATCH",
            ],
        )
    )

    fast_v1 = (
        agent.reweighted_objective_estimate(
            "repo:bug-open",
            "FAST_PATCH",
        )
    )
    robust_v1 = (
        agent.reweighted_objective_estimate(
            "repo:bug-open",
            "ROBUST_PATCH",
        )
    )

    shift = (
        agent.supersede_objective_profile(
            task_progress_weight=0.10,
            correctness_weight=0.70,
            execution_cost_weight=0.10,
            reversibility_weight=0.05,
            user_acceptance_weight=0.05,
            observed_at=1,
            reason=(
                "correctness is now more important "
                "than execution cost"
            ),
        )
    )

    current_context = (
        agent.belief_contexts.current_id
    )
    v2_scalar_state = (
        agent.objective_scalar_state_key(
            "repo:bug-open"
        )
    )

    fast_v2_prediction = (
        agent.predict_outcome(
            "repo:bug-open",
            "FAST_PATCH",
        )
    )
    robust_v2_prediction = (
        agent.predict_outcome(
            "repo:bug-open",
            "ROBUST_PATCH",
        )
    )

    shifted_choice = (
        agent.choose_action_preference_aware(
            "repo:bug-open",
            [
                "FAST_PATCH",
                "ROBUST_PATCH",
            ],
        )
    )

    robust_q_before = (
        agent.decision_policy.count(
            v2_scalar_state,
            "ROBUST_PATCH",
            belief_context_id="ctx-0",
        )
    )

    actual = (
        agent.record_objective_experience(
            shifted_choice.decision_id,
            objective_outcome=ROBUST_PATCH,
            success=True,
        )
    )

    robust_q_after = (
        agent.decision_policy.count(
            v2_scalar_state,
            "ROBUST_PATCH",
            belief_context_id="ctx-0",
        )
    )

    old_profile_prediction = (
        agent.predict_outcome(
            "repo:bug-open",
            "FAST_PATCH",
            objective_profile_reference=(
                "developer_preferences@v1"
            ),
        )
    )

    checkpoint = PROJECT_ROOT / "runtime" / "cognitive_checkpoint_v2_28_profile_demo.agentckpt"
    agent.save_checkpoint(
        checkpoint
    )
    restored = (
        core.IntegratedCognitiveAgent
        .load_checkpoint(
            checkpoint
        )
    )
    restarted_choice = (
        restored.choose_action_preference_aware(
            "repo:bug-open",
            [
                "FAST_PATCH",
                "ROBUST_PATCH",
            ],
        )
    )

    checks = {
        "initial_cost_profile_selects_fast":
            initial_choice.selected_action
            == "FAST_PATCH",
        "initial_vector_reweight_prefers_fast":
            fast_v1["utility"]
            > robust_v1["utility"],
        "preference_shift_does_not_change_belief_context":
            initial_context
            == current_context
            == "ctx-0",
        "profile_version_advanced":
            agent.objective_profile.instance_id
            == "developer_preferences@v2",
        "new_profile_scalar_world_model_starts_fresh":
            fast_v2_prediction.sample_count
            == 0
            and robust_v2_prediction.sample_count
            == 0,
        "success_history_survives_profile_shift":
            fast_v2_prediction.success_sample_count
            == 12
            and robust_v2_prediction.success_sample_count
            == 12,
        "vector_history_is_immediately_reweighted":
            (
                robust_v2_prediction
                .reweighted_objective_utility
                >
                fast_v2_prediction
                .reweighted_objective_utility
            ),
        "preference_aware_cold_start_selects_robust":
            shifted_choice.selected_action
            == "ROBUST_PATCH",
        "reweight_decision_does_not_pretrain_q":
            robust_q_before == 0,
        "actual_v2_experience_starts_v2_q_learning":
            robust_q_after == 1,
        "old_profile_scalar_history_remains_queryable":
            old_profile_prediction.sample_count
            == 12,
        "restart_preserves_new_preference":
            (
                restored.objective_profile.instance_id
                == "developer_preferences@v2"
                and restarted_choice.selected_action
                    == "ROBUST_PATCH"
            ),
    }

    result = {
        "initial_profile":
            shift["previous_profile"],
        "current_profile":
            shift["current_profile"],
        "belief_context": {
            "before": initial_context,
            "after": current_context,
        },
        "initial_reweighted_utility": {
            "FAST_PATCH":
                fast_v1["utility"],
            "ROBUST_PATCH":
                robust_v1["utility"],
        },
        "current_prediction": {
            "FAST_PATCH": {
                "scalar_reward":
                    fast_v2_prediction.predicted_reward,
                "scalar_samples":
                    fast_v2_prediction.sample_count,
                "success_probability":
                    fast_v2_prediction.predicted_success_probability,
                "success_samples":
                    fast_v2_prediction.success_sample_count,
                "reweighted_utility":
                    fast_v2_prediction.reweighted_objective_utility,
            },
            "ROBUST_PATCH": {
                "scalar_reward":
                    robust_v2_prediction.predicted_reward,
                "scalar_samples":
                    robust_v2_prediction.sample_count,
                "success_probability":
                    robust_v2_prediction.predicted_success_probability,
                "success_samples":
                    robust_v2_prediction.success_sample_count,
                "reweighted_utility":
                    robust_v2_prediction.reweighted_objective_utility,
            },
        },
        "selected": {
            "before_shift":
                initial_choice.selected_action,
            "after_shift_before_v2_learning":
                shifted_choice.selected_action,
            "after_restart":
                restarted_choice.selected_action,
        },
        "v2_q": {
            "count_before_actual":
                robust_q_before,
            "count_after_actual":
                robust_q_after,
            "actual_reward":
                actual["reward"],
        },
        "checks":
            checks,
    }

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    failed = [
        name
        for name, ok
        in checks.items()
        if not ok
    ]
    print(
        f"\nFINAL: "
        f"{len(checks)-len(failed)}/{len(checks)} PASS"
    )

    if failed:
        raise AssertionError(
            failed
        )


if __name__ == "__main__":
    main()
