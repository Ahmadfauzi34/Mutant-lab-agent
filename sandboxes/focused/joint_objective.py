
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import agen_lab as core


def archive_path(name):
    f = tempfile.NamedTemporaryFile(
        prefix=name,
        suffix=".sqlite3",
        delete=False,
    )
    f.close()
    return f.name


def record(agent, action, outcomes):
    for outcome in outcomes:
        agent.record_world_model_outcome(
            "repo",
            action,
            reward=None,
            success=True,
            objective_outcome=outcome,
        )


def missingness_demo():
    profile = core.ObjectiveUtilityProfile(
        profile_id="missingness_demo",
        task_progress_weight=0.80,
        correctness_weight=0.20,
        execution_cost_weight=0.0,
        reversibility_weight=0.0,
        user_acceptance_weight=0.0,
    )
    agent = core.IntegratedCognitiveAgent(
        "missingness_demo",
        4,
        4,
        objective_profile=profile,
        epistemic_archive_path=archive_path(
            "v228_missing_"
        ),
    )

    fragmented = [
        {"task_progress": 1.0},
        {"task_progress": 0.8},
        {"correctness": 0.0},
        {"correctness": 0.2},
    ]
    stable = [
        {
            "task_progress": 0.6,
            "correctness": 0.6,
        }
        for _ in range(4)
    ]

    record(
        agent,
        "A_FRAGMENTED",
        fragmented,
    )
    record(
        agent,
        "B_STABLE",
        stable,
    )

    fragmented_exact = (
        agent.reweighted_objective_estimate(
            "repo",
            "A_FRAGMENTED",
        )
    )
    fragmented_naive = (
        core.ObjectiveUtilityAggregator(
            agent.objective_profile
        ).aggregate(
            fragmented_exact[
                "predicted_objectives"
            ]
        ).scalar_utility
    )
    stable_exact = (
        agent.reweighted_objective_estimate(
            "repo",
            "B_STABLE",
        )
    )

    decision = (
        agent.choose_action_preference_aware(
            "repo",
            [
                "A_FRAGMENTED",
                "B_STABLE",
            ],
        )
    )

    return (
        agent,
        {
            "fragmented_naive_marginal":
                fragmented_naive,
            "fragmented_exact":
                fragmented_exact,
            "stable_exact":
                stable_exact,
            "selected":
                decision.selected_action,
        },
    )


def covariance_demo():
    profile = core.ObjectiveUtilityProfile(
        profile_id="covariance_demo",
        task_progress_weight=0.50,
        correctness_weight=0.50,
        execution_cost_weight=0.0,
        reversibility_weight=0.0,
        user_acceptance_weight=0.0,
    )
    agent = core.IntegratedCognitiveAgent(
        "covariance_demo",
        4,
        4,
        objective_profile=profile,
        epistemic_archive_path=archive_path(
            "v228_cov_"
        ),
    )

    positive = [
        {
            "task_progress": 1.0,
            "correctness": 1.0,
        },
        {
            "task_progress": 0.0,
            "correctness": 0.0,
        },
    ]
    negative = [
        {
            "task_progress": 1.0,
            "correctness": 0.0,
        },
        {
            "task_progress": 0.0,
            "correctness": 1.0,
        },
    ]

    record(
        agent,
        "POS_CORR",
        positive,
    )
    record(
        agent,
        "NEG_CORR",
        negative,
    )

    pos_prediction = (
        agent.predict_outcome(
            "repo",
            "POS_CORR",
        )
    )
    neg_prediction = (
        agent.predict_outcome(
            "repo",
            "NEG_CORR",
        )
    )

    pos_joint = (
        agent.joint_objective_model
        .group_statistics(
            "repo",
            "POS_CORR",
            (
                "task_progress",
                "correctness",
            ),
            "ctx-0",
        )
    )
    neg_joint = (
        agent.joint_objective_model
        .group_statistics(
            "repo",
            "NEG_CORR",
            (
                "task_progress",
                "correctness",
            ),
            "ctx-0",
        )
    )

    return {
        "positive": {
            "marginal_means":
                pos_prediction.predicted_objectives,
            "utility_mean":
                pos_prediction.reweighted_objective_utility,
            "utility_std":
                pos_prediction.reweighted_objective_std,
            "covariance":
                pos_joint[
                    "covariances"
                ][
                    (
                        "task_progress",
                        "correctness",
                    )
                ],
        },
        "negative": {
            "marginal_means":
                neg_prediction.predicted_objectives,
            "utility_mean":
                neg_prediction.reweighted_objective_utility,
            "utility_std":
                neg_prediction.reweighted_objective_std,
            "covariance":
                neg_joint[
                    "covariances"
                ][
                    (
                        "task_progress",
                        "correctness",
                    )
                ],
        },
    }


def main():
    (
        missing_agent,
        missing,
    ) = missingness_demo()

    covariance = (
        covariance_demo()
    )

    checkpoint = PROJECT_ROOT / "runtime" / "cognitive_checkpoint_v2_28_joint_demo.agentckpt"
    missing_agent.save_checkpoint(
        checkpoint
    )
    restored = (
        core.IntegratedCognitiveAgent
        .load_checkpoint(
            checkpoint
        )
    )
    restored_fragmented = (
        restored.reweighted_objective_estimate(
            "repo",
            "A_FRAGMENTED",
        )
    )

    checks = {
        "naive_missingness_estimate_is_0_74":
            abs(
                missing[
                    "fragmented_naive_marginal"
                ]
                - 0.74
            ) < 1e-12,
        "exact_missingness_estimate_is_0_50":
            abs(
                missing[
                    "fragmented_exact"
                ]["utility"]
                - 0.50
            ) < 1e-12,
        "stable_action_exact_is_0_60":
            abs(
                missing[
                    "stable_exact"
                ]["utility"]
                - 0.60
            ) < 1e-12,
        "exact_decision_selects_stable":
            missing["selected"]
            == "B_STABLE",
        "covariance_actions_have_same_marginal_means":
            covariance[
                "positive"
            ][
                "marginal_means"
            ] == covariance[
                "negative"
            ][
                "marginal_means"
            ],
        "covariance_actions_have_same_utility_mean":
            abs(
                covariance[
                    "positive"
                ]["utility_mean"]
                - covariance[
                    "negative"
                ]["utility_mean"]
            ) < 1e-12,
        "positive_covariance_is_positive":
            covariance[
                "positive"
            ]["covariance"] > 0,
        "negative_covariance_is_negative":
            covariance[
                "negative"
            ]["covariance"] < 0,
        "covariance_changes_utility_variance":
            covariance[
                "positive"
            ]["utility_std"]
            > covariance[
                "negative"
            ]["utility_std"],
        "restart_preserves_exact_missingness_distribution":
            (
                abs(
                    restored_fragmented[
                        "utility"
                    ]
                    - missing[
                        "fragmented_exact"
                    ]["utility"]
                ) < 1e-12
                and abs(
                    restored_fragmented[
                        "variance"
                    ]
                    - missing[
                        "fragmented_exact"
                    ]["variance"]
                ) < 1e-12
            ),
    }

    result = {
        "missingness_demo":
            missing,
        "covariance_demo":
            covariance,
        "restored_fragmented":
            restored_fragmented,
        "checks":
            checks,
    }

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=str,
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
