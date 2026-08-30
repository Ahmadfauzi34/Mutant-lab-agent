"""
Cognitive Game E2E Sandbox V2.32

End-to-end behavioral integration gate. Hidden regime labels are never placed
in policy/cognitive state. The agent must adapt from actual door outcomes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import agen_lab as core

CLAIM_KEY_REQUIRED = "door_requires_key"
SOURCE_DOOR = "door_actual_observation"
SOURCE_BASELINE_2 = "door_baseline_observation_2"
EXPECTED_REQUIRED = ["RIGHT", "PICK_KEY", "RIGHT", "OPEN_DOOR", "RIGHT"]
EXPECTED_FORBIDDEN = ["RIGHT", "RIGHT", "OPEN_DOOR", "RIGHT"]


@dataclass
class GameState:
    position: int = 0
    has_key: bool = False
    door_open: bool = False
    done: bool = False


class HiddenDoorGame:
    KEY_REQUIRED = "KEY_REQUIRED"
    KEY_FORBIDDEN = "KEY_FORBIDDEN"

    def __init__(self, regime: str):
        if regime not in (self.KEY_REQUIRED, self.KEY_FORBIDDEN):
            raise ValueError(f"unknown hidden regime: {regime}")
        self._hidden_regime = regime
        self.state = GameState()

    def set_hidden_regime(self, regime: str):
        if regime not in (self.KEY_REQUIRED, self.KEY_FORBIDDEN):
            raise ValueError(f"unknown hidden regime: {regime}")
        self._hidden_regime = regime

    def reset(self):
        self.state = GameState()

    def raw_state(self, representation_flip: bool) -> str:
        s = self.state
        if representation_flip:
            return f"door={int(s.door_open)}|key={int(s.has_key)}|position={s.position}"
        return f"position={s.position}|key={int(s.has_key)}|door={int(s.door_open)}"

    def candidates(self) -> List[str]:
        s = self.state
        if s.done:
            return []
        if s.position == 0:
            return ["RIGHT"]
        if s.position == 1 and not s.has_key:
            return ["PICK_KEY", "RIGHT"]
        if s.position == 1 and s.has_key:
            return ["RIGHT"]
        if s.position == 2 and not s.door_open:
            return ["OPEN_DOOR"]
        if s.position == 2 and s.door_open:
            return ["RIGHT"]
        return []

    def _door_opens(self) -> bool:
        if self._hidden_regime == self.KEY_REQUIRED:
            return self.state.has_key
        return not self.state.has_key

    def step(self, action: str) -> Dict:
        s = self.state
        if action not in self.candidates():
            raise ValueError(f"invalid action {action} at {s}")

        previous_position = s.position
        door_attempt = False
        door_success = None
        had_key_at_door = None

        if action == "RIGHT":
            if s.position == 0:
                s.position = 1
            elif s.position == 1:
                s.position = 2
            elif s.position == 2 and s.door_open:
                s.position = 3
                s.done = True
            else:
                raise RuntimeError("RIGHT transition not defined")
        elif action == "PICK_KEY":
            s.has_key = True
        elif action == "OPEN_DOOR":
            door_attempt = True
            had_key_at_door = bool(s.has_key)
            door_success = self._door_opens()
            if door_success:
                s.door_open = True
            else:
                s.done = True

        reached_goal = s.position == 3 and s.done

        if action == "PICK_KEY":
            objective = {
                "task_progress": 0.20,
                "correctness": 0.80,
                "execution_cost": 0.15,
                "reversibility": 1.0,
            }
            execution_success = True
        elif action == "OPEN_DOOR":
            if door_success:
                objective = {
                    "task_progress": 0.90,
                    "correctness": 1.0,
                    "execution_cost": 0.10,
                    "reversibility": 0.90,
                }
                execution_success = True
            else:
                objective = {
                    "task_progress": 0.0,
                    "correctness": 0.0,
                    "execution_cost": 0.20,
                    "reversibility": 0.90,
                }
                execution_success = False
        elif reached_goal:
            objective = {
                "task_progress": 1.0,
                "correctness": 1.0,
                "execution_cost": 0.05,
                "reversibility": 1.0,
            }
            execution_success = True
        else:
            progress = 0.30 if previous_position == 0 else 0.45
            objective = {
                "task_progress": progress,
                "correctness": 1.0,
                "execution_cost": 0.10,
                "reversibility": 1.0,
            }
            execution_success = True

        return {
            "action": action,
            "execution_success": execution_success,
            "objective_outcome": objective,
            "done": bool(s.done),
            "reached_goal": reached_goal,
            "door_attempt": door_attempt,
            "door_success": door_success,
            "had_key_at_door": had_key_at_door,
        }


def build_agent(archive_path: str):
    profile = core.ObjectiveUtilityProfile(
        profile_id="game_task_objective",
        task_progress_weight=0.65,
        correctness_weight=0.20,
        execution_cost_weight=0.10,
        reversibility_weight=0.05,
        user_acceptance_weight=0.0,
    )
    agent = core.IntegratedCognitiveAgent(
        "cognitive_game_e2e",
        4,
        4,
        epistemic_archive_path=archive_path,
        objective_profile=profile,
    )
    agent.register_source(
        core.SourceProfile(name=SOURCE_DOOR, alpha=19.0, beta=1.0)
    )
    agent.register_source(
        core.SourceProfile(name=SOURCE_BASELINE_2, alpha=19.0, beta=1.0)
    )
    return agent


def register_game_state_aliases(agent):
    for position in range(3):
        for has_key in (False, True):
            for door_open in (False, True):
                canonical = (
                    f"game:p{position}:key{int(has_key)}:door{int(door_open)}"
                )
                normal = (
                    f"position={position}|key={int(has_key)}|door={int(door_open)}"
                )
                flipped = (
                    f"door={int(door_open)}|key={int(has_key)}|position={position}"
                )
                agent.register_state_equivalence(
                    canonical,
                    f"game-state-schema-v1:{position}:{int(has_key)}:{int(door_open)}",
                    aliases=(normal, flipped),
                )


def establish_initial_key_required_belief(agent):
    agent.add_contextual_evidence(
        evidence_id="door-baseline-1",
        source=SOURCE_DOOR,
        origin_id="baseline-observation-1",
        claim_id=CLAIM_KEY_REQUIRED,
        polarity=1,
        strength=1.0,
        observed_at=0,
        observation_quality=1.0,
    )
    agent.add_contextual_evidence(
        evidence_id="door-baseline-2",
        source=SOURCE_BASELINE_2,
        origin_id="baseline-observation-2",
        claim_id=CLAIM_KEY_REQUIRED,
        polarity=1,
        strength=1.0,
        observed_at=0,
        observation_quality=1.0,
    )
    return agent.observe_claim(
        CLAIM_KEY_REQUIRED,
        notes="initial empirical door behavior",
        observed_at=0,
    )


def door_polarity_from_actual(*, had_key: bool, door_success: bool) -> int:
    supports_required = (
        (had_key and door_success)
        or ((not had_key) and (not door_success))
    )
    return 1 if supports_required else -1


class GameHarness:
    def __init__(self, agent, game: HiddenDoorGame):
        self.agent = agent
        self.game = game
        self._door_observation_counter = 0
        self.shift_events: List[Dict] = []
        self.episode_traces: List[Dict] = []

    def _observe_door_actual(
        self,
        *,
        had_key: bool,
        door_success: bool,
        observed_at: int,
    ):
        self._door_observation_counter += 1
        polarity = door_polarity_from_actual(
            had_key=had_key,
            door_success=door_success,
        )
        origin = f"door-actual-{self._door_observation_counter}"

        shift = self.agent.consider_context_shift(
            CLAIM_KEY_REQUIRED,
            incoming_polarity=polarity,
            observed_at=observed_at,
            reason="persistent actual door behavior contradicts previous requirement",
            incoming_strength=1.0,
            source=SOURCE_DOOR,
            origin_id=origin,
            observation_quality=1.0,
        )
        self.shift_events.append(dict(shift))

        self.agent.add_contextual_evidence(
            evidence_id=f"door-evidence-{self._door_observation_counter}",
            source=SOURCE_DOOR,
            origin_id=origin,
            claim_id=CLAIM_KEY_REQUIRED,
            polarity=polarity,
            strength=1.0,
            observed_at=observed_at,
            observation_quality=1.0,
        )
        self.agent.observe_claim(
            CLAIM_KEY_REQUIRED,
            notes="actual door outcome",
            observed_at=observed_at,
        )
        return shift

    def inject_low_quality_flaky_contradiction(self, observed_at: int) -> Dict:
        origin = "door-flaky-telemetry-1"
        shift = self.agent.consider_context_shift(
            CLAIM_KEY_REQUIRED,
            incoming_polarity=-1,
            observed_at=observed_at,
            reason="low-quality telemetry glitch",
            incoming_strength=1.0,
            source=SOURCE_DOOR,
            origin_id=origin,
            observation_quality=0.10,
        )
        self.shift_events.append(dict(shift))
        self.agent.add_contextual_evidence(
            evidence_id="door-flaky-evidence-1",
            source=SOURCE_DOOR,
            origin_id=origin,
            claim_id=CLAIM_KEY_REQUIRED,
            polarity=-1,
            strength=1.0,
            observed_at=observed_at,
            observation_quality=0.10,
        )
        return shift

    def run_episode(self, episode_index: int, *, max_steps: int = 8) -> Dict:
        self.game.reset()
        actions: List[str] = []
        door_shift_records = []
        start_context = self.agent.belief_contexts.current_id

        for step_index in range(max_steps):
            raw_context = self.game.raw_state(
                representation_flip=((episode_index + step_index) % 2 == 1)
            )
            candidates = self.game.candidates()
            if not candidates:
                break

            decision = self.agent.choose_action(
                raw_context,
                candidates,
                epistemic_scores={action: 0.0 for action in candidates},
            )
            prediction = self.agent.predict_outcome(
                raw_context,
                decision.selected_action,
                belief_context_id=decision.belief_context_id,
            )
            actions.append(decision.selected_action)
            actual = self.game.step(decision.selected_action)
            observed_at = self.agent.advance_interaction_clock(1)

            if actual["door_attempt"]:
                shift = self._observe_door_actual(
                    had_key=actual["had_key_at_door"],
                    door_success=bool(actual["door_success"]),
                    observed_at=observed_at,
                )
                door_shift_records.append(shift)

            done = bool(actual["done"])
            next_raw = None
            next_candidates = None
            if not done:
                next_raw = self.game.raw_state(
                    representation_flip=(
                        (episode_index + step_index + 1) % 2 == 1
                    )
                )
                next_candidates = self.game.candidates()

            self.agent.record_objective_experience(
                decision.decision_id,
                objective_outcome=actual["objective_outcome"],
                success=actual["execution_success"],
                prediction_id=prediction.prediction_id,
                next_context=next_raw,
                next_actions=next_candidates,
                done=done,
                next_belief_context_id=(
                    self.agent.belief_contexts.current_id if not done else None
                ),
            )
            if done:
                break

        trace = {
            "episode": episode_index,
            "start_belief_context": start_context,
            "end_belief_context": self.agent.belief_contexts.current_id,
            "route": list(actions),
            "reached_goal": bool(
                self.game.state.position == 3 and self.game.state.done
            ),
            "has_key": bool(self.game.state.has_key),
            "door_shift_records": door_shift_records,
        }
        self.episode_traces.append(trace)
        return trace


def run_full_scenario(*, checkpoint_path: Optional[Path] = None) -> Dict:
    archive = tempfile.NamedTemporaryFile(
        prefix="cognitive_game_v229_", suffix=".sqlite3", delete=False
    )
    archive.close()

    agent = build_agent(archive.name)
    register_game_state_aliases(agent)
    baseline_episode = establish_initial_key_required_belief(agent)

    game = HiddenDoorGame(HiddenDoorGame.KEY_REQUIRED)
    harness = GameHarness(agent, game)

    required_traces = [
        harness.run_episode(index) for index in range(1, 31)
    ]
    stable_required_routes = [item["route"] for item in required_traces[-5:]]

    noise_clock = agent.advance_interaction_clock(1)
    flaky_shift = harness.inject_low_quality_flaky_contradiction(noise_clock)
    context_after_flaky = agent.belief_contexts.current_id

    # Silent environment change. No hidden-regime label is sent to the agent.
    game.set_hidden_regime(HiddenDoorGame.KEY_FORBIDDEN)
    forbidden_traces = [
        harness.run_episode(index) for index in range(31, 51)
    ]
    stable_forbidden_routes = [item["route"] for item in forbidden_traces[-5:]]

    all_shifted = [event for event in harness.shift_events if event.get("shifted")]
    actual_negative_events = [
        event
        for event in harness.shift_events
        if event.get("incoming_polarity") == -1
        and event.get("observation_quality") == 1.0
    ]
    first_two_actual_negative = actual_negative_events[:2]

    ctx0_report = agent.adjudicate_claim(
        CLAIM_KEY_REQUIRED,
        context_id="ctx-0",
        as_of=agent.interaction_clock,
        audit_mode=core.EvidenceAuditMode.COMPACT,
    )
    ctx1_report = agent.adjudicate_claim(
        CLAIM_KEY_REQUIRED,
        context_id="ctx-1",
        as_of=agent.interaction_clock,
        audit_mode=core.EvidenceAuditMode.COMPACT,
    )

    state_alias_probe = agent.resolve_state_identity("door=0|key=0|position=1")
    required_q = agent.decision_state(
        "position=1|key=0|door=0", belief_context_id="ctx-0"
    )
    forbidden_q = agent.decision_state(
        "door=0|key=0|position=1", belief_context_id="ctx-1"
    )
    required_world = agent.predict_outcome(
        "position=2|key=1|door=0", "OPEN_DOOR", belief_context_id="ctx-0"
    )
    forbidden_world = agent.predict_outcome(
        "door=0|key=0|position=2", "OPEN_DOOR", belief_context_id="ctx-1"
    )

    if checkpoint_path is None:
        fd, temp_name = tempfile.mkstemp(
            prefix="cognitive_game_v229_", suffix=".agentckpt"
        )
        Path(temp_name).unlink(missing_ok=True)
        checkpoint = Path(temp_name)
    else:
        checkpoint = checkpoint_path
    agent.save_checkpoint(checkpoint)

    result = {
        "core_version": core.CORE_VERSION,
        "baseline_claim_admission": baseline_episode.admission_status.value,
        "required": {
            "episodes": len(required_traces),
            "last_five_routes": stable_required_routes,
            "last_five_success": [x["reached_goal"] for x in required_traces[-5:]],
        },
        "flaky_observation": {
            "detector_decision": flaky_shift["detector_decision"],
            "shifted": flaky_shift["shifted"],
            "effective_signal_strength": flaky_shift["effective_signal_strength"],
            "context_after": context_after_flaky,
        },
        "forbidden": {
            "episodes": len(forbidden_traces),
            "first_five_routes": [x["route"] for x in forbidden_traces[:5]],
            "last_five_routes": stable_forbidden_routes,
            "last_five_success": [x["reached_goal"] for x in forbidden_traces[-5:]],
        },
        "context_shift": {
            "current_context": agent.belief_contexts.current_id,
            "shift_count": len(all_shifted),
            "first_two_actual_negative": [
                {
                    "pending": event["pending"],
                    "shifted": event["shifted"],
                    "decision": event["detector_decision"],
                    "previous_context": event["previous_context"],
                    "current_context": event["current_context"],
                }
                for event in first_two_actual_negative
            ],
        },
        "belief_reports": {
            "ctx-0": {
                "evidence_status": (ctx0_report["evidence_status"].value if hasattr(ctx0_report["evidence_status"], "value") else str(ctx0_report["evidence_status"])),
                "support": ctx0_report["support_score"],
                "oppose": ctx0_report["oppose_score"],
            },
            "ctx-1": {
                "evidence_status": (ctx1_report["evidence_status"].value if hasattr(ctx1_report["evidence_status"], "value") else str(ctx1_report["evidence_status"])),
                "support": ctx1_report["support_score"],
                "oppose": ctx1_report["oppose_score"],
            },
        },
        "state_identity": {
            "raw": "door=0|key=0|position=1",
            "canonical": state_alias_probe.canonical_id,
        },
        "q_snapshot": {
            "ctx-0": required_q["q_values"],
            "ctx-1": forbidden_q["q_values"],
        },
        "world_model": {
            "required_open_door": {
                "scalar_samples": required_world.sample_count,
                "success_samples": required_world.success_sample_count,
                "success_probability": required_world.predicted_success_probability,
                "objective_support": required_world.reweighted_objective_support,
            },
            "forbidden_open_door": {
                "scalar_samples": forbidden_world.sample_count,
                "success_samples": forbidden_world.success_sample_count,
                "success_probability": forbidden_world.predicted_success_probability,
                "objective_support": forbidden_world.reweighted_objective_support,
            },
        },
        "memory": {
            "decision_records": len(agent.decision_memory.all()),
            "prediction_records": len(agent.prediction_memory.all()),
            "prediction_error_records": len(agent.prediction_error_memory.all()),
            "joint_objective_groups": len(agent.joint_objective_model._groups),
        },
        "checkpoint": str(checkpoint),
    }

    checks = {
        "core_is_v2_32": core.CORE_VERSION == "2.42",
        "hidden_regime_not_in_any_policy_state": all(
            "KEY_REQUIRED" not in str(key) and "KEY_FORBIDDEN" not in str(key)
            for key in agent.decision_policy.scoped_q_values.keys()
        ),
        "initial_empirical_claim_stays_pending_not_grounded": (
            baseline_episode.admission_status == core.AdmissionStatus.PENDING
        ),
        "required_route_converges": all(
            route == EXPECTED_REQUIRED for route in stable_required_routes
        ),
        "required_last_five_success": all(
            x["reached_goal"] for x in required_traces[-5:]
        ),
        "low_quality_noise_does_not_shift": (
            not flaky_shift["shifted"] and context_after_flaky == "ctx-0"
        ),
        "persistent_real_change_opens_exactly_one_context": (
            len(all_shifted) == 1
            and agent.belief_contexts.current_id == "ctx-1"
        ),
        "first_real_contradiction_pending": (
            len(first_two_actual_negative) >= 2
            and first_two_actual_negative[0]["pending"]
            and not first_two_actual_negative[0]["shifted"]
        ),
        "second_real_contradiction_confirms_shift": (
            len(first_two_actual_negative) >= 2
            and first_two_actual_negative[1]["shifted"]
            and first_two_actual_negative[1]["current_context"] == "ctx-1"
        ),
        "forbidden_route_converges": all(
            route == EXPECTED_FORBIDDEN for route in stable_forbidden_routes
        ),
        "forbidden_last_five_success": all(
            x["reached_goal"] for x in forbidden_traces[-5:]
        ),
        "old_context_history_preserved": ctx0_report["support_score"] > 0.0,
        "new_context_learns_opposite_empirical_direction": (
            ctx1_report["oppose_score"] > ctx1_report["support_score"]
        ),
        "state_aliases_reuse_one_canonical_identity": (
            state_alias_probe.canonical_id == "game:p1:key0:door0"
        ),
        "q_is_belief_context_scoped": (
            required_q["belief_context_id"] == "ctx-0"
            and forbidden_q["belief_context_id"] == "ctx-1"
        ),
        "world_model_has_actual_samples_in_both_contexts": (
            required_world.success_sample_count > 0
            and forbidden_world.success_sample_count > 0
        ),
        "prediction_calibration_received_actual_feedback": (
            len(agent.prediction_error_memory.all()) > 0
        ),
        "joint_objective_model_received_actual_experience": (
            len(agent.joint_objective_model._groups) > 0
        ),
    }
    result["checks"] = checks
    return result


def run_restart_probe(checkpoint_path: Path):
    agent = core.IntegratedCognitiveAgent.load_checkpoint(checkpoint_path)
    game = HiddenDoorGame(HiddenDoorGame.KEY_FORBIDDEN)
    harness = GameHarness(agent, game)
    trace = harness.run_episode(1001)
    result = {
        "core_version": core.CORE_VERSION,
        "belief_context": agent.belief_contexts.current_id,
        "route": trace["route"],
        "reached_goal": trace["reached_goal"],
        "profile": agent.objective_profile.instance_id,
        "state_registry_entries": agent.state_registry.state()["canonical_count"],
    }
    print("RESTART_RESULT_JSON=" + json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-probe", type=Path, default=None)
    args = parser.parse_args()

    if args.restart_probe is not None:
        run_restart_probe(args.restart_probe)
        return

    fd, temp_name = tempfile.mkstemp(
        prefix="cognitive_game_v229_restart_", suffix=".agentckpt"
    )
    checkpoint_file = Path(temp_name)
    checkpoint_file.unlink(missing_ok=True)

    result = run_full_scenario(checkpoint_path=checkpoint_file)

    probe = subprocess.run(
        [sys.executable, str(Path(__file__)), "--restart-probe", str(checkpoint_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    restart_line = next(
        (
            line
            for line in probe.stdout.splitlines()
            if line.startswith("RESTART_RESULT_JSON=")
        ),
        None,
    )
    restart_result = (
        json.loads(restart_line.split("=", 1)[1]) if restart_line else {}
    )
    restart_checks = {
        "fresh_process_returncode": probe.returncode == 0,
        "restart_keeps_ctx1": restart_result.get("belief_context") == "ctx-1",
        "restart_keeps_forbidden_route_policy": (
            restart_result.get("route") == EXPECTED_FORBIDDEN
        ),
        "restart_episode_reaches_goal": restart_result.get("reached_goal") is True,
        "restart_core_version": restart_result.get("core_version") == "2.42",
    }

    result["fresh_process_restart"] = {
        "result": restart_result,
        "checks": restart_checks,
    }
    all_checks = dict(result["checks"])
    all_checks.update(restart_checks)
    result["all_checks"] = all_checks

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    failed = [name for name, ok in all_checks.items() if not ok]
    print(f"\nFINAL: {len(all_checks)-len(failed)}/{len(all_checks)} PASS")

    checkpoint_file.unlink(missing_ok=True)
    for sidecar in checkpoint_file.parent.glob(checkpoint_file.name + ".*.cold.sqlite3"):
        sidecar.unlink(missing_ok=True)

    if failed:
        print("\nFAILED CHECKS:")
        for name in failed:
            print("-", name)
        raise AssertionError(failed)


if __name__ == "__main__":
    main()
