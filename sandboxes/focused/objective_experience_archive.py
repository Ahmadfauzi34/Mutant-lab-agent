
"""V2.30 focused sandbox — exact COLD objective experience replay."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agen_lab as core


def main():
    with tempfile.TemporaryDirectory(
        prefix="v230_objexp_sandbox_"
    ) as tmp:
        root = Path(tmp)

        agent = core.IntegratedCognitiveAgent(
            "developer-objective-audit",
            4,
            4,
            epistemic_archive_path=str(
                root / "cold.sqlite3"
            ),
        )

        # Adversarial missingness history.
        for outcome in (
            {"task_progress": 1.0},
            {"task_progress": 0.8},
            {"correctness": 0.0},
            {"correctness": 0.2},
        ):
            agent.record_world_model_outcome(
                "repo:working",
                "PATCH",
                None,
                True,
                objective_outcome=outcome,
            )

        archived_before_shift = (
            agent.epistemic_archive
            .objective_experience_count()
        )
        old_profile_ids = {
            record.scalarization_profile_instance_id
            for record in (
                agent.epistemic_archive
                .objective_experiences()
            )
        }

        agent.supersede_objective_profile(
            task_progress_weight=0.8,
            correctness_weight=0.2,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=1,
            reason="developer priority shift",
        )

        learning_before = {
            "q":
                copy.deepcopy(
                    agent.decision_policy
                    .scoped_counts
                ),
            "scalar":
                copy.deepcopy(
                    agent.contextual_world_model
                    ._stats
                ),
            "joint":
                copy.deepcopy(
                    agent.joint_objective_model
                    ._groups
                ),
            "marginal":
                copy.deepcopy(
                    agent.objective_world_model
                    ._stats
                ),
            "success":
                copy.deepcopy(
                    agent.success_constraint_model
                    ._stats
                ),
            "archive":
                agent.epistemic_archive
                .objective_experience_count(),
        }

        replay = (
            agent.replay_objective_experience_utility(
                "repo:working",
                "PATCH",
                include_records=True,
            )
        )

        learning_after = {
            "q":
                copy.deepcopy(
                    agent.decision_policy
                    .scoped_counts
                ),
            "scalar":
                copy.deepcopy(
                    agent.contextual_world_model
                    ._stats
                ),
            "joint":
                copy.deepcopy(
                    agent.joint_objective_model
                    ._groups
                ),
            "marginal":
                copy.deepcopy(
                    agent.objective_world_model
                    ._stats
                ),
            "success":
                copy.deepcopy(
                    agent.success_constraint_model
                    ._stats
                ),
            "archive":
                agent.epistemic_archive
                .objective_experience_count(),
        }

        # Compare positive vs negative covariance with equal means.
        profile = agent.supersede_objective_profile(
            task_progress_weight=0.5,
            correctness_weight=0.5,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=2,
            reason="equal weights covariance audit",
        )

        for outcome in (
            {
                "task_progress": 1.0,
                "correctness": 1.0,
            },
            {
                "task_progress": 0.0,
                "correctness": 0.0,
            },
        ):
            agent.record_world_model_outcome(
                "repo:risk",
                "VOLATILE",
                None,
                True,
                objective_outcome=outcome,
            )

        for outcome in (
            {
                "task_progress": 1.0,
                "correctness": 0.0,
            },
            {
                "task_progress": 0.0,
                "correctness": 1.0,
            },
        ):
            agent.record_world_model_outcome(
                "repo:risk",
                "STABLE",
                None,
                True,
                objective_outcome=outcome,
            )

        volatile = (
            agent.replay_objective_experience_utility(
                "repo:risk",
                "VOLATILE",
            )
        )
        stable = (
            agent.replay_objective_experience_utility(
                "repo:risk",
                "STABLE",
            )
        )

        # Q-only structured feedback is intentionally outside the objective
        # world archive.
        q_only_before = (
            agent.epistemic_archive
            .objective_experience_count()
        )
        decision = agent.choose_action(
            "repo:q-only",
            ["LOCAL_EDIT"],
        )
        agent.record_decision_outcome(
            decision.decision_id,
            objective_outcome={
                "task_progress": 0.9,
            },
        )
        q_only_after = (
            agent.epistemic_archive
            .objective_experience_count()
        )

        state = agent.epistemic_archive_state()

        checks = {
            "core_v2_30":
                core.CORE_VERSION
                == "2.42",
            "four_missingness_records_archived":
                archived_before_shift
                == 4,
            "preference_shift_does_not_duplicate_history":
                agent.epistemic_archive
                .objective_experience_count()
                == 8,
            "old_record_profile_provenance_retained":
                len(old_profile_ids)
                == 1
                and next(
                    iter(
                        old_profile_ids
                    )
                ).endswith(
                    "@v1"
                ),
            "exact_missingness_mean_is_point_five":
                abs(
                    replay["mean"]
                    - 0.5
                )
                <= 1e-12,
            "replay_and_joint_stats_agree":
                replay[
                    "joint_agreement"
                ][
                    "exact_within_tolerance"
                ],
            "record_level_history_complete":
                replay[
                    "record_level_history"
                ][
                    "completeness"
                ]
                == "complete",
            "replay_is_nonlearning":
                learning_before
                == learning_after,
            "equal_means_different_variance":
                abs(
                    volatile["mean"]
                    - stable["mean"]
                )
                <= 1e-12
                and volatile["variance"]
                    > stable["variance"],
            "stable_variance_zero":
                abs(
                    stable["variance"]
                )
                <= 1e-12,
            "q_only_feedback_not_in_objective_world_archive":
                q_only_before
                == q_only_after,
            "objective_archive_is_not_evidence":
                state["cold"][
                    "objective_experiences"
                ]
                == 8
                and state["cold"][
                    "evidence"
                ]
                == 0,
        }

        result = {
            "core_version":
                core.CORE_VERSION,
            "replay":
                replay,
            "volatile":
                volatile,
            "stable":
                stable,
            "archive_state":
                state,
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
            "\nFINAL: "
            f"{len(checks)-len(failed)}"
            f"/{len(checks)} PASS"
        )

        if failed:
            print(
                "FAILED:",
                failed,
            )
            raise AssertionError(
                failed
            )


if __name__ == "__main__":
    main()
