"""V2.30 focused sandbox: preference-aware risk + trajectory integration.

This is an integration laboratory, not a new architecture identity.
It demonstrates that exact reweighted actual-vector history can influence
risk/trajectory ranking without becoming current-profile Q experience.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agen_lab as core


def profile(profile_id="sandbox-pref"):
    return core.ObjectiveUtilityProfile(
        profile_id=profile_id,
        task_progress_weight=0.5,
        correctness_weight=0.5,
        execution_cost_weight=0.0,
        reversibility_weight=0.0,
        user_acceptance_weight=0.0,
    )


def agent_in(root: Path):
    return core.IntegratedCognitiveAgent(
        "v229-preference-risk-sandbox",
        4,
        4,
        epistemic_archive_path=str(root / "cold.sqlite3"),
        objective_profile=profile(),
    )


def covariance_pair(agent, state, volatile, stable, repeats=6):
    for _ in range(repeats):
        for outcome in (
            {"task_progress": 1.0, "correctness": 1.0},
            {"task_progress": 0.0, "correctness": 0.0},
        ):
            agent.record_world_model_outcome(
                state, volatile, None, True, objective_outcome=outcome
            )
        for outcome in (
            {"task_progress": 1.0, "correctness": 0.0},
            {"task_progress": 0.0, "correctness": 1.0},
        ):
            agent.record_world_model_outcome(
                state, stable, None, True, objective_outcome=outcome
            )


def same_weight_supersession(agent, observed_at=1):
    return agent.supersede_objective_profile(
        task_progress_weight=0.5,
        correctness_weight=0.5,
        execution_cost_weight=0.0,
        reversibility_weight=0.0,
        user_acceptance_weight=0.0,
        observed_at=observed_at,
        reason="same utility semantics, explicit new preference lifecycle",
    )


def learning_snapshot(agent):
    return {
        "q_counts": copy.deepcopy(agent.decision_policy.scoped_counts),
        "scalar_world": copy.deepcopy(agent.contextual_world_model._stats),
        "joint": copy.deepcopy(agent.joint_objective_model._groups),
        "success": copy.deepcopy(agent.success_constraint_model._stats),
        "prediction_count": len(agent.prediction_memory.all()),
    }


def main():
    with tempfile.TemporaryDirectory(prefix="v229_pref_risk_sandbox_") as tmp:
        root = Path(tmp)
        agent = agent_in(root)

        # Same marginal utility mean, radically different covariance/volatility.
        covariance_pair(agent, "repo:s1", "VOLATILE", "STABLE", repeats=6)
        same_weight_supersession(agent)

        volatile = agent.preference_aware_utility_estimate("repo:s1", "VOLATILE")
        stable = agent.preference_aware_utility_estimate("repo:s1", "STABLE")

        before_rank = learning_snapshot(agent)
        conservative = agent.choose_action_preference_aware_risk(
            "repo:s1",
            ["VOLATILE", "STABLE"],
            requested_mode=core.RiskMode.CONSERVATIVE,
        )
        after_rank = learning_snapshot(agent)

        # Coverage: same utility mean, but PART is only half scorable under the
        # new task-only preference. Coverage is policy uncertainty, not bad utility.
        for _ in range(12):
            agent.record_world_model_outcome(
                "repo:coverage", "FULL", None, True,
                objective_outcome={"task_progress": 0.5},
            )
        for _ in range(6):
            agent.record_world_model_outcome(
                "repo:coverage", "PART", None, True,
                objective_outcome={"task_progress": 0.5},
            )
            agent.record_world_model_outcome(
                "repo:coverage", "PART", None, True,
                objective_outcome={"correctness": 0.5},
            )
        agent.supersede_objective_profile(
            task_progress_weight=1.0,
            correctness_weight=0.0,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=2,
            reason="task-only preference",
        )
        full = agent.preference_aware_utility_estimate("repo:coverage", "FULL")
        part = agent.preference_aware_utility_estimate("repo:coverage", "PART")
        coverage_choice = agent.choose_action_preference_aware_risk(
            "repo:coverage",
            ["FULL", "PART"],
            requested_mode=core.RiskMode.CONSERVATIVE,
        )

        # Exploratory policy can use bounded information bonus only when the
        # independent technical gate says the action is safe enough. Build the
        # mixed-component history while both components are scorable, then
        # reinterpret it under task-only preference.
        agent.supersede_objective_profile(
            task_progress_weight=0.5,
            correctness_weight=0.5,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=3,
            reason="balanced history collection",
        )
        for _ in range(12):
            agent.record_world_model_outcome(
                "repo:explore", "KNOWN", None, True,
                objective_outcome={"task_progress": 0.55},
            )
        for _ in range(6):
            agent.record_world_model_outcome(
                "repo:explore", "UNCERTAIN", None, True,
                objective_outcome={"task_progress": 0.55},
            )
            agent.record_world_model_outcome(
                "repo:explore", "UNCERTAIN", None, True,
                objective_outcome={"correctness": 0.55},
            )
        agent.supersede_objective_profile(
            task_progress_weight=1.0,
            correctness_weight=0.0,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=4,
            reason="task-only exploration preference",
        )
        exploratory = agent.choose_action_preference_aware_risk(
            "repo:explore",
            ["KNOWN", "UNCERTAIN"],
            failure_consequences={"KNOWN": 0.1, "UNCERTAIN": 0.1},
            requested_mode=core.RiskMode.EXPLORATORY,
        )

        # High utility + high uncertainty is still blocked if actual technical
        # execution history is known-bad.
        for _ in range(12):
            agent.record_world_model_outcome(
                "repo:gate", "BAD", None, False,
                objective_outcome={"task_progress": 1.0},
            )
            agent.record_world_model_outcome(
                "repo:gate", "SAFE", None, True,
                objective_outcome={"task_progress": 0.5},
            )
        gated = agent.choose_action_preference_aware_risk(
            "repo:gate",
            ["BAD", "SAFE"],
            requested_mode=core.RiskMode.EXPLORATORY,
        )

        # Two-state trajectory: same expected utility, stable covariance wins.
        covariance_pair(agent, "repo:t1", "V1", "S1", repeats=6)
        covariance_pair(agent, "repo:t2", "V2", "S2", repeats=6)
        # Re-open equal-weight preference to make the covariance contrast exact.
        agent.supersede_objective_profile(
            task_progress_weight=0.5,
            correctness_weight=0.5,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=5,
            reason="trajectory balanced preference",
        )
        before_trajectory = learning_snapshot(agent)
        trajectory = agent.choose_trajectory_preference_aware_risk(
            {
                "STABLE_PATH": (("repo:t1", "S1"), ("repo:t2", "S2")),
                "VOLATILE_PATH": (("repo:t1", "V1"), ("repo:t2", "V2")),
            },
            requested_mode=core.RiskMode.BALANCED,
        )
        after_trajectory = learning_snapshot(agent)

        stale = agent.preference_aware_utility_estimate("repo:t1", "S1")
        agent.supersede_objective_profile(
            task_progress_weight=0.5,
            correctness_weight=0.5,
            execution_cost_weight=0.0,
            reversibility_weight=0.0,
            user_acceptance_weight=0.0,
            observed_at=6,
            reason="new lifecycle with same semantic weights",
        )
        stale_rejected = False
        try:
            agent.validate_preference_aware_utility_estimate(stale)
        except core.ObjectiveProfileVersionConflict:
            stale_rejected = True

        # Actual experience under the new profile must supersede cold-start
        # reweighting for that exact profile via current-profile Q.
        cold = agent.preference_aware_utility_estimate("repo:q", "PATCH")
        decision_bundle = agent.choose_action_preference_aware_risk(
            "repo:q", ["PATCH"], requested_mode=core.RiskMode.BALANCED
        )
        agent.record_objective_experience(
            decision_bundle["decision"].decision_id,
            {"task_progress": 0.8, "correctness": 0.8},
            success=True,
        )
        warm = agent.preference_aware_utility_estimate("repo:q", "PATCH")

        checks = {
            "core_v2_30": core.CORE_VERSION == "2.42",
            "equal_mean_covariance_distinguished": (
                abs(volatile.mean - stable.mean) < 1e-12
                and volatile.aleatoric_std > stable.aleatoric_std
            ),
            "conservative_prefers_lower_volatility": (
                conservative["ranking"]["selected_action"] == "STABLE"
            ),
            "action_ranking_is_read_only_learning": before_rank == after_rank,
            "low_coverage_is_not_low_utility": (
                abs(full.mean - part.mean) < 1e-12
                and full.coverage == 1.0
                and part.coverage == 0.5
            ),
            "conservative_coverage_policy_prefers_full": (
                coverage_choice["ranking"]["selected_action"] == "FULL"
            ),
            "exploratory_can_choose_uncertain_safe_action": (
                exploratory["ranking"]["selected_action"] == "UNCERTAIN"
            ),
            "technical_known_bad_gate_overrides_bonus": (
                gated["ranking"]["selected_action"] == "SAFE"
                and "BAD" in gated["ranking"]["blocked_actions"]
            ),
            "trajectory_prefers_stable_path": (
                trajectory["ranking"]["selected_trajectory_id"] == "STABLE_PATH"
            ),
            "trajectory_ranking_is_read_only_learning": (
                before_trajectory == after_trajectory
            ),
            "same_weights_new_profile_invalidates_stale_estimate": stale_rejected,
            "actual_new_profile_experience_promotes_q_source": (
                cold.source == "neutral_prior"
                and warm.source == "profile_q"
                and warm.q_sample_count == 1
            ),
        }

        result = {
            "core_version": core.CORE_VERSION,
            "covariance_demo": {
                "volatile": {
                    "mean": volatile.mean,
                    "std": volatile.aleatoric_std,
                    "support": volatile.support,
                },
                "stable": {
                    "mean": stable.mean,
                    "std": stable.aleatoric_std,
                    "support": stable.support,
                },
                "conservative_selected": conservative["ranking"]["selected_action"],
            },
            "coverage_demo": {
                "full_mean": full.mean,
                "part_mean": part.mean,
                "full_coverage": full.coverage,
                "part_coverage": part.coverage,
                "selected": coverage_choice["ranking"]["selected_action"],
            },
            "exploration_demo": {
                "selected": exploratory["ranking"]["selected_action"],
                "uncertain_bonus": exploratory["ranking"]["action_audit"]["UNCERTAIN"]["information_bonus"],
            },
            "technical_gate_demo": {
                "selected": gated["ranking"]["selected_action"],
                "blocked": list(gated["ranking"]["blocked_actions"]),
            },
            "trajectory_demo": {
                "selected": trajectory["ranking"]["selected_trajectory_id"],
                "stable_is_experience": trajectory["ranking"]["trajectory_audit"]["STABLE_PATH"]["counterfactual_is_experience"],
            },
            "checks": checks,
        }

        print(json.dumps(result, indent=2, sort_keys=True))
        failed = [name for name, ok in checks.items() if not ok]
        print(f"\nFINAL: {len(checks) - len(failed)}/{len(checks)} PASS")
        if failed:
            print("FAILED:")
            for name in failed:
                print("-", name)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
