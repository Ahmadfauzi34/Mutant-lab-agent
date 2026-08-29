"""Integrated cognitive-agent orchestration — physical extraction M7.

This module owns the real V2.38 ``IntegratedCognitiveAgent`` implementation.
Subsystem implementation is imported directly from physical modules; this file
does not depend on ``agen_lab._kernel`` or the canonical compatibility shim.

Trusted-local pickle compatibility is retained by assigning the class
serialized identity to ``agen_kognitif_v2_28`` after definition.
"""
from __future__ import annotations

import math
import uuid

from dataclasses import replace
from typing import Dict, List, Set, Tuple, Optional

CORE_VERSION = "2.42"
INTEGRATION_CANDIDATE = "V3.0-V3.2_PORT_ON_V2.24_OBSERVATION_RELIABILITY"
ONTOLOGY_WEIGHT = 0.5

from .planning import (
    Domain, KnowledgeBase, Point, SpaceTimeNode,
    SpatioTemporalCostmap, SpatioTemporalPlanner, RouteAction,
    SAFETY_CLEARANCE, COLLISION_MARGIN,
)
from .memory import (
    BeliefShiftDecisionMemory, EpisodeMemory, TransitionMemory, DecisionMemory,
    TrajectoryDecisionMemory, CounterfactualMemory, MetaRiskDecisionMemory,
    PredictionMemory, PredictionErrorMemory, MemoryRetentionPolicy,
    MemoryLifecycleManager, EpistemicArchivePolicy, EpistemicArchiveManager,
    ObjectiveExperienceRecord,
)
from .objectives import (
    OBJECTIVE_COMPONENTS, ObjectiveOutcome, ObjectiveUtilityProfile,
    ObjectiveProfileVersionConflict, ObjectiveProfileRegistry,
    ObjectiveAggregation, ObjectiveUtilityAggregator, ContextScopedObjectiveModel,
    ContextScopedJointObjectiveModel, ContextScopedSuccessConstraintModel,
)
from .identity import (
    StateCanonicalDefinition, ResolvedStateIdentity, StateIdentityConflict,
    StateIdentityRegistry, ActionDefinition, ActionVersionConflict,
    ActionRegistry, ResolvedActionIdentity,
)
from .epistemic import (
    SourceProfile, Evidence, BeliefContext, BeliefContextManager,
    BeliefShiftDecisionRecord, ContextualBeliefRevisionPolicy, SourceLineage,
    EvidenceAggregator, GroundedFact, GroundingStore, Rule, RuleVersionConflict,
    RuleValidator, Justification, TruthEvaluator, EpistemicVerdict, AuditReport,
    AdmissionStatus, Episode, KnowledgeAdmissionPolicy, TruthMaintenanceSystem,
    EvidenceAuditMode, ExactEvidenceQueryEngine,
)
from .decision import (
    DecisionRecord, TransitionRecord, TrajectoryDecisionRecord, DecisionPolicy,
    UncertaintyDecisionMode, UncertaintyRiskProfile, MetaRiskDecision,
    AdaptiveRiskModePolicy, RiskMode, ActionRiskEstimate,
    ChanceConstrainedSafetyGate, MetaRiskPolicy, UncertaintyAwareDecisionPolicy,
    TrajectoryRiskEstimate, TrajectoryChanceConstrainedSafetyGate,
    TrajectoryRiskPolicy, PreferenceAwareUtilityEstimate,
    PreferenceAwareRiskPolicy, PreferenceAwareTrajectoryEstimate,
    PreferenceAwareTrajectoryRiskPolicy,
)
from .world_model import (
    WorldOutcome, WorldDecisionEpisode, WorldOutcomeEvaluator,
    CounterfactualEstimate, StrategyExecution, PredictionErrorRecord,
    PredictionUncertaintyEstimator, OutcomePrediction, ContextScopedWorldModel,
    WorldModelReliability, PredictionErrorEvaluator, CounterfactualStrategyPolicy,
)
from .persistence import (
    PERSISTENCE_SCHEMA_VERSION, AgentPersistenceError, AgentPersistenceManager,
)
from .portable_state import (
    PORTABLE_STATE_SCHEMA_VERSION, PortableStateError, PortableStateTypeError,
    PortableCognitiveStateManager,
)
from .patterns import (
    PatternError, PatternSourceConflict, PatternPredictionError,
    PatternKind, PatternRelationType, StructuralPatternCandidate,
    StructuralPatternDefinition, StructuralPatternHypothesis,
    StructuralPatternInstance, StructuralPatternPrediction,
    StructuralPatternPredictionAssessment, PatternRelation,
    StructuralPatternEngine, StructuralPatternStore,
    canonicalize_symbol_sequence,
)
from .spatial import (
    MAX_SPATIAL_OBJECTS_PER_SCENE, DEFAULT_SPATIAL_SCENE_LIMIT,
    MAX_SPATIAL_RELATIONS, SpatialError, SpatialSceneConflict,
    SpatialRelationConflict, SpatialRelationType, SpatialRelationSource,
    SpatialPose2D, SpatialExtent2D, SpatialBounds2D, SpatialObject2D,
    SpatialScene2D, SpatialRelation, SpatialRelationAlgebra,
    SpatialGeometry2D, SpatialSceneCanonicalizer, SpatialSceneStore,
    make_spatial_scene,
)
from .spatial_transform import (
    DEFAULT_TRANSFORM_MATCH_TOLERANCE, MAX_TRANSFORM_MATCH_OBJECTS,
    SpatialTransformError, SpatialTransformFrameError,
    SpatialTransformMatchError, SpatialLinearTransformKind,
    SpatialTransform2D, SpatialTransformMatch, SpatialTransformInference,
    SpatialTransformationMatcher, spatial_transform_token,
)
from .spatial_manipulation import (
    MAX_MANIPULATION_SCENE_OBJECTS, SpatialManipulationError,
    SpatialManipulationKind, SpatialManipulationCheckKind,
    SpatialManipulationOperator, SpatialManipulationCheck,
    SpatialManipulationCollision, CounterfactualSpatialManipulation,
    SpatialManipulationSimulator, spatial_manipulation_token,
)
from .spatial_planning import (
    DEFAULT_SPATIAL_PLAN_MAX_DEPTH, DEFAULT_SPATIAL_PLAN_MAX_NODES,
    DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS, MAX_SPATIAL_PLAN_OPERATOR_CATALOG,
    SpatialPlanningError, SpatialPlanningStatus, SpatialRelationGoal,
    SpatialManipulationPlanStep, SpatialManipulationPlan,
    SpatialManipulationPlanningResult, BoundedSpatialManipulationPlanner,
    spatial_plan_token,
)
from .spatial_execution import (
    DEFAULT_SPATIAL_EXECUTION_TICKET_LIMIT,
    DEFAULT_SPATIAL_EXECUTION_MATCH_TOLERANCE,
    SpatialExecutionError, SpatialExecutionConflict,
    SpatialExecutionStaleSource, SpatialExecutionContinuationBlocked,
    SpatialExecutionTicketStatus, SpatialExecutionFeedbackStatus,
    SpatialExecutionTicket, SpatialExecutionFeedback,
    SpatialExecutionComparator, SpatialExecutionStore,
)
from .spatial_replanning import (
    DEFAULT_SPATIAL_REPLAN_RECORD_LIMIT,
    SpatialReplanningError, SpatialReplanningConflict,
    SpatialReplanningTriggerStatus, SpatialReplanningRecord,
    SpatialReplanningStore, DeviationTriggeredSpatialReplanner,
)
from .spatial_recovery import (
    DEFAULT_SPATIAL_RECOVERY_RECORD_LIMIT,
    DEFAULT_SPATIAL_RECOVERY_MAX_HANDOFF_STEPS,
    SPATIAL_RECOVERY_POLICY_VERSION, SpatialRecoveryError,
    SpatialRecoveryConflict, SpatialRecoveryAction, SpatialRecoveryReason,
    SpatialRecoveryDecisionRecord, DeterministicSpatialRecoveryPolicy,
    SpatialRecoveryStore,
)
from .spatial_reliability import (
    DEFAULT_SPATIAL_RELIABILITY_STAT_LIMIT,
    DEFAULT_SPATIAL_RELIABILITY_UPDATE_LIMIT,
    DEFAULT_SPATIAL_RELIABILITY_ASSESSMENT_LIMIT,
    DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES,
    DEFAULT_SPATIAL_RELIABILITY_WILSON_Z,
    DEFAULT_SPATIAL_RECOVERY_RELIABILITY_THRESHOLD,
    SpatialReliabilityError, SpatialReliabilityConflict,
    SpatialReliabilityGateBlocked, SpatialReliabilityAggregationLevel,
    SpatialReliabilityUpdateDisposition, SpatialRecoveryReliabilityStatus,
    SpatialReliabilityAccumulator, SpatialReliabilityUpdate,
    SpatialReliabilityEstimate, SpatialPlanReliabilityStep,
    SpatialRecoveryReliabilityAssessment,
    SpatialManipulationReliabilityStore,
)
from .spatial_plan_ranking import (
    DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
    DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
    SpatialPlanRankingError, SpatialPlanRankingConflict,
    SpatialPlanReliabilityRankingStatus, SpatialPlanReliabilityCandidate,
    SpatialReliabilityRankedPlanningResult, SpatialPlanReliabilityRanker,
)
from .spatial_replan_ranking import (
    DEFAULT_SPATIAL_REPLAN_RANKING_MIN_SAMPLES,
    DEFAULT_SPATIAL_REPLAN_RANKING_WILSON_Z,
    SpatialReplanRankingError, SpatialReplanRankingConflict,
    SpatialReliabilityRankedReplanView, SpatialReplanReliabilityRanker,
)

class IntegratedCognitiveAgent:
    def __init__(
        self,
        domain_name: str,
        width: int,
        height: int,
        retention_policy: Optional[
            MemoryRetentionPolicy
        ] = None,
        epistemic_archive_policy: Optional[
            EpistemicArchivePolicy
        ] = None,
        epistemic_archive_path: Optional[
            str
        ] = None,
        belief_revision_policy: Optional[
            ContextualBeliefRevisionPolicy
        ] = None,
        objective_profile: Optional[
            ObjectiveUtilityProfile
        ] = None,
    ):
        self.domain = Domain(domain_name)
        self.kb = KnowledgeBase(self.domain)
        self.rule_validator = RuleValidator(self.domain)
        self.truth_evaluator = TruthEvaluator(self.rule_validator)

        self.sources: Dict[str, SourceProfile] = {}
        self.evidence_aggregator = EvidenceAggregator(self.sources)

        self.costmap = SpatioTemporalCostmap(width, height)
        self.planner = SpatioTemporalPlanner(self.costmap)

        self.grounded: Set[str] = set()
        # Legacy global provenance/current compatibility.
        self.grounded_provenance: Dict[str, Dict] = {}

        # V2.10 — scoped grounding registry.
        self.grounding_store = GroundingStore()

        self.justifications: List[Justification] = []
        self.evidence_pool: List[Evidence] = []

        # V2.9 — temporal/contextual belief state
        self.belief_contexts = BeliefContextManager()

        # V2.18 — monotonic clock of ACTUAL domain interactions.
        # This is distinct in meaning from a belief-context identifier and is
        # persisted so adapters can restart without timestamp regression.
        self.interaction_clock = 0

        # V2.22 — action/tool implementation identity is orthogonal to
        # belief-context identity.
        self.action_registry = ActionRegistry()

        # V2.25 — state equivalence is explicit and orthogonal to both
        # belief-context and action identity.
        self.state_registry = StateIdentityRegistry()

        self.belief_revision_policy = (
            belief_revision_policy
            if belief_revision_policy is not None
            else ContextualBeliefRevisionPolicy()
        )
        self.belief_shift_memory = (
            BeliefShiftDecisionMemory(
                limit=2048
            )
        )
        self._belief_shift_decision_counter = 0

        self.contextual_admission: Dict[
            Tuple[str, str], AdmissionStatus
        ] = {}

        # V2.2 — learning state
        self.memory = EpisodeMemory()
        self.admission_policy = KnowledgeAdmissionPolicy()
        self.accepted_claims: Set[str] = set()
        self.pending_claims: Set[str] = set()
        self.quarantined_claims: Set[str] = set()
        self.rejected_claims: Set[str] = set()
        self._episode_counter = 0

        # V2.3 — explicit truth maintenance
        self.tms = TruthMaintenanceSystem(self)

        # V2.4 — decision learning state
        self.decision_policy = DecisionPolicy()
        self.decision_memory = DecisionMemory()
        self.trajectory_decision_memory = TrajectoryDecisionMemory()
        self.transition_memory = TransitionMemory()
        self._decision_counter = 0
        self._trajectory_decision_counter = 0
        self._transition_counter = 0

        # V2.5 — world/outcome coupling
        self.world_outcome_evaluator = WorldOutcomeEvaluator()
        self.world_decision_history: List[WorldDecisionEpisode] = []

        # V2.6 — counterfactual / strategy layer
        self.counterfactual_policy = CounterfactualStrategyPolicy()
        self.counterfactual_memory = CounterfactualMemory()
        self._counterfactual_counter = 0

        # V2.7 — prediction-error / world-model calibration
        self.world_model_reliability = WorldModelReliability()
        self.prediction_error_evaluator = PredictionErrorEvaluator()
        self.prediction_error_memory = PredictionErrorMemory()
        self._prediction_error_counter = 0

        # V2.13 — context-scoped empirical world model + forecast memory
        self.contextual_world_model = ContextScopedWorldModel()

        # V2.26 — structured objective semantics.
        initial_objective_profile = (
            objective_profile
            if objective_profile is not None
            else ObjectiveUtilityProfile()
        )
        self.objective_profile_registry = (
            ObjectiveProfileRegistry()
        )
        self.objective_profile = (
            self.objective_profile_registry.register(
                initial_objective_profile,
                activated_at=0,
            )
        )

        # The initial profile deliberately uses the exact V2.26 scalar keys.
        # Later profile versions get isolated scalar namespaces.
        self._objective_compatibility_instance_id = (
            self.objective_profile.instance_id
        )

        self.objective_aggregator = (
            ObjectiveUtilityAggregator(
                self.objective_profile
            )
        )
        self.objective_world_model = (
            ContextScopedObjectiveModel()
        )
        self.joint_objective_model = (
            ContextScopedJointObjectiveModel()
        )
        self.success_constraint_model = (
            ContextScopedSuccessConstraintModel()
        )

        self.prediction_memory = PredictionMemory()
        self.prediction_uncertainty_estimator = (
            PredictionUncertaintyEstimator(
                confidence_level=0.95,
                minimum_sufficient_samples=10,
            )
        )
        self.uncertainty_decision_policy = (
            UncertaintyAwareDecisionPolicy()
        )

        # V3.1/V3.2 chance gates share one action threshold by construction.
        self.chance_safety_gate = (
            self.uncertainty_decision_policy.safety_gate
        )
        self.meta_risk_policy = (
            self.uncertainty_decision_policy.meta_policy
        )
        self.trajectory_chance_safety_gate = (
            TrajectoryChanceConstrainedSafetyGate(
                max_trajectory_failure_probability=0.25,
                action_gate=self.chance_safety_gate,
            )
        )
        self.trajectory_risk_policy = TrajectoryRiskPolicy(
            safety_gate=self.trajectory_chance_safety_gate,
        )

        # V2.29 — preference-aware risk consumes exact reweighted actual-vector
        # uncertainty read-only. The independent technical gate is shared with
        # the existing action chance constraint by construction.
        self.preference_aware_risk_policy = PreferenceAwareRiskPolicy(
            safety_gate=self.chance_safety_gate,
        )
        self.preference_aware_trajectory_risk_policy = (
            PreferenceAwareTrajectoryRiskPolicy(
                safety_gate=self.chance_safety_gate,
            )
        )

        # V2.16 — adaptive risk/meta-decision.
        self.adaptive_risk_mode_policy = (
            AdaptiveRiskModePolicy()
        )
        self.meta_risk_memory = (
            MetaRiskDecisionMemory()
        )
        self._meta_decision_counter = 0

        # V2.18 — predictions that still participate in an in-flight causal
        # action/outcome cycle cannot be compacted until assessed.
        self._active_prediction_pins: Set[int] = set()

        self._outcome_prediction_counter = 0

        # V2.17 — bounded operational-memory lifecycle.
        self.memory_lifecycle = MemoryLifecycleManager(
            self,
            retention_policy
            if retention_policy is not None
            else MemoryRetentionPolicy(),
        )

        # V2.19 — exact cold Evidence/Episode archive.
        self.epistemic_archive_policy = (
            epistemic_archive_policy
            if epistemic_archive_policy is not None
            else EpistemicArchivePolicy()
        )
        self.epistemic_archive = (
            EpistemicArchiveManager(
                epistemic_archive_path
            )
        )
        self.memory.archive_manager = (
            self.epistemic_archive
        )

        # V2.20 — exact indexed adjudication performance state.
        self._evidence_revision = 0
        self.evidence_query_engine = (
            ExactEvidenceQueryEngine(
                self,
                cache_limit=256,
            )
        )

        # V2.32 — exact symbolic structural-pattern cognition.
        # This is a bounded, context-scoped hypothesis/reliability subsystem.
        # It is intentionally orthogonal to Evidence, Q/TD, world models, and
        # Belief Context revision.
        self.structural_patterns = StructuralPatternStore()

        # V2.33 — bounded object-centric 2D spatial state.
        # Spatial state is a model representation, not Evidence/truth, and is
        # deliberately isolated from Q/world-model/pattern support unless an
        # adapter explicitly routes derived symbolic tokens elsewhere.
        self.spatial_scenes = SpatialSceneStore()

        # V2.37 — bounded execution/feedback journal. This does not actuate
        # the world. It binds external dispatch receipts and ACTUAL submitted
        # spatial observations to exact counterfactual plan-step predictions.
        self.spatial_execution = SpatialExecutionStore()

        # V2.38 — bounded durable journal of explicit deviation-triggered
        # replanning attempts. Replanning itself remains counterfactual and
        # never mutates original plan/feedback provenance.
        self.spatial_replanning = SpatialReplanningStore()

        # V2.39 — deterministic recovery-policy journal. The policy may
        # recommend continue/replan/abort/intervention/handoff, but never
        # replans, dispatches, or physically executes as a side effect.
        self.spatial_recovery = SpatialRecoveryStore()

        # V2.40 — empirical manipulation reliability learned only from CLOSED
        # actual execution feedback. It is orthogonal to Q/world/Evidence and
        # supplies read-only recovery confidence / explicit gated handoff.
        self.spatial_reliability = SpatialManipulationReliabilityStore()

    # -----------------------------------------------------------------
    # V2.41 — RELIABILITY-AWARE EQUAL-DEPTH PLAN RANKING
    # -----------------------------------------------------------------

    def rank_spatial_planning_result_by_reliability(
        self,
        planning_result: SpatialManipulationPlanningResult,
        *,
        min_samples: int = DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
        ranked_at: Optional[int] = None,
    ) -> SpatialReliabilityRankedPlanningResult:
        stamp = self.interaction_clock if ranked_at is None else int(ranked_at)
        # Read-only over V2.40 reliability. Ranking does not move interaction
        # time because no actual external event occurred.
        return SpatialPlanReliabilityRanker.rank(
            planning_result,
            self.spatial_reliability,
            min_samples=min_samples,
            wilson_z=wilson_z,
            ranked_at=stamp,
        )

    def plan_spatial_manipulation_reliability_aware(
        self,
        scene_id: str,
        goal: SpatialRelationGoal,
        operators,
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
        min_samples: int = DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
        ranked_at: Optional[int] = None,
    ) -> SpatialReliabilityRankedPlanningResult:
        base = self.plan_spatial_manipulation(
            scene_id,
            goal,
            operators,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
        )
        return self.rank_spatial_planning_result_by_reliability(
            base,
            min_samples=min_samples,
            wilson_z=wilson_z,
            ranked_at=ranked_at,
        )

    def plan_spatial_manipulation_on_scene_reliability_aware(
        self,
        scene: SpatialScene2D,
        goal: SpatialRelationGoal,
        operators,
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
        min_samples: int = DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
        ranked_at: Optional[int] = None,
    ) -> SpatialReliabilityRankedPlanningResult:
        base = self.plan_spatial_manipulation_on_scene(
            scene,
            goal,
            operators,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
        )
        return self.rank_spatial_planning_result_by_reliability(
            base,
            min_samples=min_samples,
            wilson_z=wilson_z,
            ranked_at=ranked_at,
        )

    # -----------------------------------------------------------------
    # RELIABILITY-RANKED REPLAN VIEW — DERIVED / READ-ONLY
    # -----------------------------------------------------------------

    def rank_spatial_replan_by_reliability(
        self,
        replan_id: str,
        *,
        min_samples: int = DEFAULT_SPATIAL_REPLAN_RANKING_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_REPLAN_RANKING_WILSON_Z,
        ranked_at: Optional[int] = None,
    ) -> SpatialReliabilityRankedReplanView:
        replan_record = self.spatial_replanning.get(replan_id)
        stamp = self.interaction_clock if ranked_at is None else int(ranked_at)
        # Derived/read-only interpretation of an already completed replan.
        # It neither changes the immutable replan record nor advances actual
        # interaction time.
        return SpatialReplanReliabilityRanker.rank(
            replan_record,
            self.spatial_reliability,
            min_samples=min_samples,
            wilson_z=wilson_z,
            ranked_at=stamp,
        )

    # -----------------------------------------------------------------
    # V2.40 — EMPIRICAL MANIPULATION RELIABILITY / RECOVERY CONFIDENCE
    # -----------------------------------------------------------------

    def assess_spatial_recovery_reliability(
        self,
        recovery_id: str,
        *,
        min_samples: int = DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES,
        minimum_wilson_lower_bound: float = (
            DEFAULT_SPATIAL_RECOVERY_RELIABILITY_THRESHOLD
        ),
        evaluated_at: Optional[int] = None,
    ) -> SpatialRecoveryReliabilityAssessment:
        decision = self.spatial_recovery.get(recovery_id)
        if not decision.can_prepare_handoff or decision.replan_id is None:
            raise SpatialReliabilityError(
                "recovery decision tidak memiliki replacement handoff"
            )
        replan_record = self.spatial_replanning.get(decision.replan_id)
        replacement = replan_record.replacement_plan
        if replacement is None:
            raise SpatialReliabilityConflict(
                "recovery reliability kehilangan replacement plan"
            )
        stamp = self.interaction_clock if evaluated_at is None else int(evaluated_at)
        # Reliability assessment is read-only cognitive work.
        return self.spatial_reliability.assess_recovery(
            decision,
            replacement,
            min_samples=min_samples,
            minimum_wilson_lower_bound=minimum_wilson_lower_bound,
            evaluated_at=stamp,
        )

    def prepare_reliability_gated_spatial_recovery_handoff(
        self,
        recovery_id: str,
        *,
        assessment_id: Optional[str] = None,
        min_samples: int = DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES,
        minimum_wilson_lower_bound: float = (
            DEFAULT_SPATIAL_RECOVERY_RELIABILITY_THRESHOLD
        ),
        prepared_at: Optional[int] = None,
    ) -> SpatialExecutionTicket:
        assessment = (
            self.assess_spatial_recovery_reliability(
                recovery_id,
                min_samples=min_samples,
                minimum_wilson_lower_bound=minimum_wilson_lower_bound,
            )
            if assessment_id is None
            else self.spatial_reliability.get_assessment(assessment_id)
        )
        if assessment.recovery_id != recovery_id:
            raise SpatialReliabilityConflict(
                "assessment bukan milik recovery decision"
            )
        if (
            assessment.reliability_revision
            != self.spatial_reliability.reliability_revision
        ):
            raise SpatialReliabilityConflict(
                "reliability assessment stale; evaluasi ulang diperlukan"
            )
        if not assessment.trusted:
            raise SpatialReliabilityGateBlocked(
                "replacement handoff diblokir oleh empirical reliability gate: "
                + assessment.status.value
            )
        return self.prepare_spatial_recovery_handoff(
            recovery_id,
            prepared_at=prepared_at,
        )

    def spatial_reliability_update(
        self,
        feedback_id: str,
    ) -> Optional[SpatialReliabilityUpdate]:
        return self.spatial_reliability.update_for_feedback(feedback_id)

    def spatial_reliability_assessment(
        self,
        assessment_id: str,
    ) -> SpatialRecoveryReliabilityAssessment:
        return self.spatial_reliability.get_assessment(assessment_id)

    def spatial_reliability_state(self) -> Dict:
        return self.spatial_reliability.state()

    # -----------------------------------------------------------------
    # V2.39 — SPATIAL RECOVERY POLICY / CONTROLLED REPLACEMENT HANDOFF
    # -----------------------------------------------------------------

    def evaluate_spatial_recovery(
        self,
        original_plan: SpatialManipulationPlan,
        ticket_id: str,
        *,
        replan_id: Optional[str] = None,
        max_handoff_steps: int = (
            DEFAULT_SPATIAL_RECOVERY_MAX_HANDOFF_STEPS
        ),
        evaluated_at: Optional[int] = None,
    ) -> SpatialRecoveryDecisionRecord:
        ticket = self.spatial_execution.get_ticket(ticket_id)
        feedback = self.spatial_execution.feedback(ticket_id)
        if feedback is None:
            raise SpatialRecoveryConflict(
                "execution ticket belum memiliki closed actual feedback"
            )
        replan_record = (
            None
            if replan_id is None
            else self.spatial_replanning.get(replan_id)
        )
        action, reason, replacement_plan_id = (
            DeterministicSpatialRecoveryPolicy.decide(
                original_plan,
                ticket,
                feedback,
                replan_record=replan_record,
                max_handoff_steps=max_handoff_steps,
            )
        )
        stamp = (
            self.interaction_clock
            if evaluated_at is None
            else int(evaluated_at)
        )
        # Recovery evaluation is cognitive policy work, not a new actual-world
        # interaction. Do not advance interaction_clock here.
        return self.spatial_recovery.add(
            original_plan=original_plan,
            ticket=ticket,
            feedback=feedback,
            replan_record=replan_record,
            action=action,
            reason=reason,
            replacement_plan_id=replacement_plan_id,
            max_handoff_steps=max_handoff_steps,
            evaluated_at=stamp,
        )

    def prepare_spatial_recovery_handoff(
        self,
        recovery_id: str,
        *,
        prepared_at: Optional[int] = None,
    ) -> SpatialExecutionTicket:
        decision = self.spatial_recovery.get(recovery_id)
        if not decision.can_prepare_handoff:
            raise SpatialRecoveryConflict(
                "recovery decision tidak eligible untuk replacement handoff"
            )
        if decision.handoff_ticket_id is not None:
            return self.spatial_execution.get_ticket(
                decision.handoff_ticket_id
            )
        if decision.replan_id is None:
            raise SpatialRecoveryConflict(
                "handoff decision kehilangan replan provenance"
            )
        replan_record = self.spatial_replanning.get(
            decision.replan_id
        )
        replacement = replan_record.replacement_plan
        if replacement is None:
            raise SpatialRecoveryConflict(
                "handoff membutuhkan replacement plan"
            )
        if replacement.plan_id != decision.replacement_plan_id:
            raise SpatialRecoveryConflict(
                "replacement plan identity berbeda dari recovery decision"
            )
        feedback = self.spatial_execution.feedback(
            decision.trigger_ticket_id
        )
        if feedback is None:
            raise SpatialRecoveryConflict(
                "handoff kehilangan actual trigger feedback"
            )
        source_scene = feedback.observed_scene
        actual_signature = (
            SpatialSceneCanonicalizer.exact_signature(source_scene)
        )
        if actual_signature != decision.actual_scene_signature:
            raise SpatialRecoveryConflict(
                "actual scene provenance berubah sejak recovery decision"
            )
        first_step = replacement.steps[0]
        if first_step.source_scene_signature != actual_signature:
            raise SpatialRecoveryConflict(
                "replacement plan tidak berakar pada actual scene"
            )
        simulation = SpatialManipulationSimulator.simulate(
            source_scene,
            first_step.operator,
            predicted_scene_id=(
                f"recovery-handoff-{recovery_id}-step-1"
            ),
        )
        if not simulation.feasible:
            raise SpatialRecoveryConflict(
                "replacement first operator tidak lagi feasible"
            )
        stamp = (
            self.interaction_clock
            if prepared_at is None
            else int(prepared_at)
        )
        ticket = self.spatial_execution.issue(
            plan=replacement,
            step_index=1,
            source_scene=source_scene,
            predicted_scene=simulation.predicted_scene,
            prepared_at=stamp,
        )
        self.spatial_recovery.bind_handoff_ticket(
            recovery_id,
            ticket.ticket_id,
        )
        return ticket

    def spatial_recovery_record(
        self,
        recovery_id: str,
    ) -> SpatialRecoveryDecisionRecord:
        return self.spatial_recovery.get(recovery_id)

    def latest_spatial_recovery_for_plan(
        self,
        original_plan_id: str,
    ) -> Optional[SpatialRecoveryDecisionRecord]:
        return self.spatial_recovery.latest_for_plan(
            original_plan_id
        )

    def spatial_recovery_state(self) -> Dict:
        return self.spatial_recovery.state()

    # -----------------------------------------------------------------
    # V2.38 — DEVIATION-TRIGGERED BOUNDED SPATIAL REPLANNING
    # -----------------------------------------------------------------

    def replan_spatial_after_execution_deviation(
        self,
        original_plan: SpatialManipulationPlan,
        ticket_id: str,
        operators,
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
        requested_at: Optional[int] = None,
    ) -> SpatialReplanningRecord:
        ticket = self.spatial_execution.get_ticket(ticket_id)
        feedback = self.spatial_execution.feedback(ticket_id)
        if feedback is None:
            raise SpatialReplanningConflict(
                "execution ticket belum memiliki closed actual feedback"
            )

        operator_catalog = tuple(operators)
        result = DeviationTriggeredSpatialReplanner.replan(
            original_plan,
            ticket,
            feedback,
            operator_catalog,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
        )
        stamp = (
            self.interaction_clock
            if requested_at is None
            else int(requested_at)
        )
        # Replanning is cognitive/counterfactual work, not a new actual-world
        # interaction; do not advance interaction_clock here.
        return self.spatial_replanning.add(
            original_plan=original_plan,
            ticket=ticket,
            feedback=feedback,
            operators=operator_catalog,
            planning_result=result,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
            requested_at=stamp,
        )

    def spatial_replanning_record(
        self,
        replan_id: str,
    ) -> SpatialReplanningRecord:
        return self.spatial_replanning.get(replan_id)

    def latest_spatial_replan_for_plan(
        self,
        original_plan_id: str,
    ) -> Optional[SpatialReplanningRecord]:
        return self.spatial_replanning.latest_for_plan(
            original_plan_id
        )

    def spatial_replanning_state(self) -> Dict:
        return self.spatial_replanning.state()

    # -----------------------------------------------------------------
    # V2.37 — PLAN EXECUTION / ACTUAL OBSERVATION FEEDBACK BOUNDARY
    # -----------------------------------------------------------------

    def prepare_spatial_plan_execution_step(
        self,
        plan: SpatialManipulationPlan,
        step_index: int = 1,
        *,
        prepared_at: Optional[int] = None,
    ) -> SpatialExecutionTicket:
        if not isinstance(plan, SpatialManipulationPlan):
            raise SpatialExecutionError(
                "plan harus SpatialManipulationPlan"
            )
        step_index = int(step_index)
        if step_index < 1 or step_index > plan.step_count:
            raise SpatialExecutionError(
                "step_index di luar plan"
            )

        step = plan.steps[step_index - 1]
        stamp = (
            self.interaction_clock
            if prepared_at is None
            else int(prepared_at)
        )

        if step_index == 1:
            source_scene = self.spatial_scenes.get(
                plan.source_scene_id
            )
            source_signature = (
                SpatialSceneCanonicalizer.exact_signature(
                    source_scene
                )
            )
            if source_signature != step.source_scene_signature:
                raise SpatialExecutionStaleSource(
                    "registered source scene berubah sejak plan dibuat"
                )
        else:
            previous = (
                self.spatial_execution.latest_plan_feedback(
                    plan.plan_id,
                    step_index - 1,
                )
            )
            if previous is None:
                raise SpatialExecutionContinuationBlocked(
                    "step sebelumnya belum memiliki actual feedback"
                )
            if not previous.can_continue_plan:
                raise SpatialExecutionContinuationBlocked(
                    "actual feedback step sebelumnya menyimpang dari prediksi; "
                    "replanning eksplisit diperlukan"
                )
            source_scene = previous.observed_scene

        simulation = SpatialManipulationSimulator.simulate(
            source_scene,
            step.operator,
            predicted_scene_id=(
                f"exec-pred-{plan.plan_id}-{step_index}"
            ),
        )
        if not simulation.feasible:
            raise SpatialExecutionContinuationBlocked(
                "operator plan tidak lagi feasible pada actual source scene"
            )

        return self.spatial_execution.issue(
            plan=plan,
            step_index=step_index,
            source_scene=source_scene,
            predicted_scene=simulation.predicted_scene,
            prepared_at=stamp,
        )

    def acknowledge_spatial_execution_dispatch(
        self,
        ticket_id: str,
        *,
        external_receipt: str,
        dispatched_at: Optional[int] = None,
    ) -> SpatialExecutionTicket:
        stamp = (
            self.interaction_clock
            if dispatched_at is None
            else int(dispatched_at)
        )
        ticket = self.spatial_execution.dispatch(
            ticket_id,
            external_receipt=external_receipt,
            dispatched_at=stamp,
        )
        # External dispatch acknowledgment is an actual interaction boundary,
        # but still not evidence of success.
        self.touch_interaction_time(stamp)
        return ticket

    def submit_spatial_execution_observation(
        self,
        ticket_id: str,
        observed_scene: SpatialScene2D,
        *,
        observed_at: Optional[int] = None,
        tolerance: float = (
            DEFAULT_SPATIAL_EXECUTION_MATCH_TOLERANCE
        ),
        register_actual_scene: bool = True,
    ) -> SpatialExecutionFeedback:
        if not isinstance(observed_scene, SpatialScene2D):
            raise SpatialExecutionError(
                "actual observation harus SpatialScene2D"
            )

        stamp = (
            observed_scene.observed_at
            if observed_at is None
            else int(observed_at)
        )
        if (
            observed_scene.observed_at is not None
            and int(observed_scene.observed_at) != stamp
        ):
            raise SpatialExecutionConflict(
                "observed_at argument bertentangan dengan scene provenance"
            )

        ticket = self.spatial_execution.get_ticket(
            ticket_id
        )
        if ticket.status not in (
            SpatialExecutionTicketStatus.DISPATCHED,
            SpatialExecutionTicketStatus.CLOSED,
        ):
            raise SpatialExecutionConflict(
                "actual observation hanya diterima setelah external dispatch"
            )

        feedback = SpatialExecutionComparator.compare(
            ticket,
            observed_scene,
            observed_at=stamp,
            tolerance=tolerance,
        )
        feedback = (
            self.spatial_execution.close_with_feedback(
                ticket_id,
                feedback,
            )
        )

        # V2.40: only CLOSED feedback backed by an actual observation can
        # update manipulation reliability. Idempotence is keyed by feedback_id.
        self.spatial_reliability.observe_closed_feedback(
            ticket,
            feedback,
        )

        # This is the actual-observation boundary. Clock movement does not
        # imply truth/Q/world-model learning.
        self.touch_interaction_time(stamp)

        if register_actual_scene:
            try:
                self.spatial_scenes.register(
                    observed_scene
                )
            except SpatialSceneConflict:
                existing = self.spatial_scenes.get(
                    observed_scene.scene_id
                )
                if existing != observed_scene:
                    raise

        return feedback

    def cancel_spatial_execution_ticket(
        self,
        ticket_id: str,
        *,
        reason: str,
        cancelled_at: Optional[int] = None,
    ) -> SpatialExecutionTicket:
        stamp = (
            self.interaction_clock
            if cancelled_at is None
            else int(cancelled_at)
        )
        return self.spatial_execution.cancel(
            ticket_id,
            reason=reason,
            cancelled_at=stamp,
        )

    def spatial_execution_ticket(
        self,
        ticket_id: str,
    ) -> SpatialExecutionTicket:
        return self.spatial_execution.get_ticket(
            ticket_id
        )

    def spatial_execution_feedback(
        self,
        ticket_id: str,
    ) -> Optional[SpatialExecutionFeedback]:
        return self.spatial_execution.feedback(
            ticket_id
        )

    def spatial_execution_state(self) -> Dict:
        return self.spatial_execution.state()

    # -----------------------------------------------------------------
    # V2.36 — BOUNDED SPATIAL MANIPULATION PLANNING
    # -----------------------------------------------------------------

    def plan_spatial_manipulation(
        self,
        scene_id: str,
        goal: SpatialRelationGoal,
        operators,
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
    ) -> SpatialManipulationPlanningResult:
        return BoundedSpatialManipulationPlanner.search(
            self.spatial_scenes.get(scene_id),
            goal,
            operators,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
        )

    def plan_spatial_manipulation_on_scene(
        self,
        scene: SpatialScene2D,
        goal: SpatialRelationGoal,
        operators,
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
    ) -> SpatialManipulationPlanningResult:
        return BoundedSpatialManipulationPlanner.search(
            scene,
            goal,
            operators,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
        )

    def spatial_plan_token(
        self,
        plan: SpatialManipulationPlan,
    ) -> Tuple[Tuple[str, str], ...]:
        return spatial_plan_token(plan)

    # -----------------------------------------------------------------
    # V2.35 — COUNTERFACTUAL SPATIAL MANIPULATION
    # -----------------------------------------------------------------

    def simulate_spatial_manipulation(
        self,
        scene_id: str,
        operator: SpatialManipulationOperator,
        *,
        predicted_scene_id: Optional[str] = None,
    ) -> CounterfactualSpatialManipulation:
        """Simulate one manipulation without execution or learning."""
        return SpatialManipulationSimulator.simulate(
            self.spatial_scenes.get(scene_id),
            operator,
            predicted_scene_id=predicted_scene_id,
        )

    def simulate_spatial_manipulation_on_scene(
        self,
        scene: SpatialScene2D,
        operator: SpatialManipulationOperator,
        *,
        predicted_scene_id: Optional[str] = None,
    ) -> CounterfactualSpatialManipulation:
        """Stateless chaining surface for future planning layers."""
        return SpatialManipulationSimulator.simulate(
            scene,
            operator,
            predicted_scene_id=predicted_scene_id,
        )

    def spatial_manipulation_token(
        self,
        operator: SpatialManipulationOperator,
        *,
        include_object_ids: bool = False,
        include_numeric_parameters: bool = False,
    ) -> Tuple:
        return spatial_manipulation_token(
            operator,
            include_object_ids=include_object_ids,
            include_numeric_parameters=include_numeric_parameters,
        )

    # -----------------------------------------------------------------
    # V2.34 — SPATIAL TRANSFORMATION ALGEBRA
    # -----------------------------------------------------------------

    def apply_spatial_transform(
        self,
        scene_id: str,
        transform: SpatialTransform2D,
        *,
        target_scene_id: Optional[str] = None,
        observed_at: Optional[int] = None,
        register: bool = False,
    ) -> SpatialScene2D:
        """Apply an explicit D4+translation transform to one retained scene.

        The transformed scene is counterfactual/model state unless the caller
        explicitly registers it. Even registration remains spatial-state
        registration, not Evidence/Q/world-model experience.
        """
        scene = self.spatial_scenes.get(scene_id)
        transformed = transform.apply_scene(
            scene,
            scene_id=target_scene_id,
            observed_at=(
                scene.observed_at
                if observed_at is None
                else int(observed_at)
            ),
        )
        if register:
            self.spatial_scenes.register(transformed)
        return transformed

    def infer_spatial_transform(
        self,
        source_scene_id: str,
        target_scene_id: str,
        *,
        tolerance: float = DEFAULT_TRANSFORM_MATCH_TOLERANCE,
        require_labels: bool = True,
        require_namespace: bool = True,
        require_belief_context: bool = True,
    ) -> SpatialTransformInference:
        return SpatialTransformationMatcher.infer(
            self.spatial_scenes.get(source_scene_id),
            self.spatial_scenes.get(target_scene_id),
            tolerance=tolerance,
            require_labels=require_labels,
            require_namespace=require_namespace,
            require_belief_context=require_belief_context,
        )

    def compose_spatial_transforms(
        self,
        first: SpatialTransform2D,
        second: SpatialTransform2D,
    ) -> SpatialTransform2D:
        return first.then(second)

    def invert_spatial_transform(
        self,
        transform: SpatialTransform2D,
    ) -> SpatialTransform2D:
        return transform.inverse()

    def spatial_transform_token(
        self,
        transform: SpatialTransform2D,
        *,
        include_translation: bool = False,
        include_frames: bool = False,
    ) -> Tuple:
        return spatial_transform_token(
            transform,
            include_translation=include_translation,
            include_frames=include_frames,
        )

    # -----------------------------------------------------------------
    # V2.33 — OBJECT-CENTRIC SPATIAL STATE / RELATION ALGEBRA
    # -----------------------------------------------------------------

    def register_spatial_scene(
        self,
        objects,
        *,
        namespace: str = "default",
        frame_id: str = "world",
        scene_id: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        observed_at: Optional[int] = None,
    ) -> Dict:
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        stamp = (
            self.interaction_clock
            if observed_at is None
            else int(observed_at)
        )
        scene = make_spatial_scene(
            objects,
            namespace=namespace,
            belief_context_id=scope,
            frame_id=frame_id,
            scene_id=scene_id,
            observed_at=stamp,
        )
        result = self.spatial_scenes.register(scene)
        result["exact_signature"] = (
            SpatialSceneCanonicalizer.exact_signature(
                result["scene"]
            )
        )
        result["translation_normalized_signature"] = (
            SpatialSceneCanonicalizer.translation_normalized_signature(
                result["scene"]
            )
        )
        return result

    def spatial_scene(
        self,
        scene_id: str,
    ) -> SpatialScene2D:
        return self.spatial_scenes.get(scene_id)

    def infer_spatial_relations(
        self,
        scene_id: str,
    ) -> Tuple[SpatialRelation, ...]:
        scene = self.spatial_scenes.get(scene_id)
        return SpatialGeometry2D.scene_relations(scene)

    def query_spatial_relations(
        self,
        scene_id: str,
        subject_id: str,
        object_id: str,
    ) -> Tuple[SpatialRelation, ...]:
        return tuple(
            relation
            for relation in self.infer_spatial_relations(scene_id)
            if (
                relation.subject_id == subject_id
                and relation.object_id == object_id
            )
        )

    def close_spatial_relations(
        self,
        relations,
        *,
        max_relations: int = MAX_SPATIAL_RELATIONS,
    ) -> Tuple[SpatialRelation, ...]:
        return SpatialRelationAlgebra.close(
            relations,
            max_relations=max_relations,
        )

    def spatial_scene_signatures(
        self,
        scene_id: str,
    ) -> Dict:
        scene = self.spatial_scenes.get(scene_id)
        relations = SpatialGeometry2D.scene_relations(scene)
        return {
            "scene_id": scene.scene_id,
            "belief_context_id": scene.belief_context_id,
            "namespace": scene.namespace,
            "frame_id": scene.frame_id,
            "exact_signature": (
                SpatialSceneCanonicalizer.exact_signature(scene)
            ),
            "translation_normalized_signature": (
                SpatialSceneCanonicalizer.translation_normalized_signature(
                    scene
                )
            ),
            "relational_signature": (
                SpatialSceneCanonicalizer.relational_signature(
                    scene,
                    relations,
                )
            ),
            "learning_mutation": False,
        }

    def spatial_relation_tokens(
        self,
        scene_id: str,
    ) -> Tuple[Tuple[str, str, str], ...]:
        """Read-only symbolic bridge to V2.32 pattern adapters.

        Returning tokens does NOT call `observe_structural_sequence`; therefore
        spatial relation extraction itself never creates pattern support.
        """
        relations = self.infer_spatial_relations(scene_id)
        return tuple(
            sorted(
                (
                    relation.relation_type.value,
                    relation.subject_id,
                    relation.object_id,
                )
                for relation in relations
            )
        )

    def spatial_state(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        return self.spatial_scenes.state(
            namespace=namespace,
            belief_context_id=scope,
        )

    # -----------------------------------------------------------------
    # V2.32 — STRUCTURAL PATTERN REPRESENTATION / DISCOVERY
    # -----------------------------------------------------------------

    def canonicalize_structural_sequence(
        self,
        sequence,
    ) -> Tuple[str, ...]:
        return canonicalize_symbol_sequence(sequence)

    def discover_structural_patterns(
        self,
        sequence,
    ) -> Tuple[StructuralPatternCandidate, ...]:
        """Pure read-only structural discovery.

        No pattern support, prediction reliability, Evidence, Q, or world-model
        state is updated by this method.
        """
        return StructuralPatternEngine.discover(sequence)

    def observe_structural_sequence(
        self,
        sequence,
        *,
        namespace: str = "default",
        belief_context_id: Optional[str] = None,
        source_id: Optional[str] = None,
        observed_at: Optional[int] = None,
    ) -> Dict:
        """Register one actual symbolic sequence in the pattern subsystem.

        Pattern learning remains isolated from epistemic truth, Q/TD, and world
        models. `source_id` provides exact retry/dedup identity when adapters
        have one.
        """
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        stamp = (
            self.interaction_clock
            if observed_at is None
            else int(observed_at)
        )
        return self.structural_patterns.observe_sequence(
            sequence,
            namespace=namespace,
            belief_context_id=scope,
            observed_at=stamp,
            source_id=source_id,
        )

    def structural_pattern_hypotheses(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        kind: Optional[PatternKind] = None,
    ) -> Tuple[StructuralPatternHypothesis, ...]:
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        return self.structural_patterns.hypotheses(
            namespace=namespace,
            belief_context_id=scope,
            kind=kind,
        )

    def predict_structural_next(
        self,
        sequence,
        *,
        namespace: str = "default",
        belief_context_id: Optional[str] = None,
        generated_at: Optional[int] = None,
    ) -> Optional[StructuralPatternPrediction]:
        """Generate a pattern prediction without creating experience."""
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        stamp = (
            self.interaction_clock
            if generated_at is None
            else int(generated_at)
        )
        return self.structural_patterns.predict_next(
            sequence,
            namespace=namespace,
            belief_context_id=scope,
            generated_at=stamp,
        )

    def assess_structural_prediction(
        self,
        prediction_id: str,
        actual_symbol,
        *,
        assessed_at: Optional[int] = None,
    ) -> StructuralPatternPredictionAssessment:
        """Update only pattern predictive reliability from an actual result."""
        stamp = (
            self.interaction_clock
            if assessed_at is None
            else int(assessed_at)
        )
        return self.structural_patterns.assess_prediction(
            prediction_id,
            actual_symbol,
            assessed_at=stamp,
        )

    def structural_pattern_relational_completion(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        relation_types=None,
    ) -> Dict:
        return self.structural_patterns.relational_completion(
            node_id,
            max_depth=max_depth,
            relation_types=relation_types,
        )

    def structural_pattern_topology_audit(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        include_predictions: bool = False,
        hub_threshold: int = 4,
    ) -> Dict:
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        return self.structural_patterns.topology_audit(
            namespace=namespace,
            belief_context_id=scope,
            include_predictions=include_predictions,
            hub_threshold=hub_threshold,
        )

    def structural_pattern_state(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        scope = (
            self.belief_contexts.current_id
            if belief_context_id is None
            else belief_context_id
        )
        return self.structural_patterns.state(
            namespace=namespace,
            belief_context_id=scope,
        )

    # -----------------------------------------------------------------
    # V2.27 — OBJECTIVE PROFILE IDENTITY / VERSIONING
    # -----------------------------------------------------------------

    def resolve_objective_profile(
        self,
        reference: Optional[str] = None,
    ) -> ObjectiveUtilityProfile:
        if reference is None:
            return self.objective_profile

        return (
            self.objective_profile_registry
            .resolve(
                reference
            )
        )

    def objective_profile_versions(
        self,
        profile_id: Optional[str] = None,
    ) -> List[ObjectiveUtilityProfile]:
        family = (
            self.objective_profile.profile_id
            if profile_id is None
            else profile_id
        )
        return (
            self.objective_profile_registry
            .all_versions(
                family
            )
        )

    def _archive_actual_objective_experience(
        self,
        *,
        context: str,
        belief_context_id: Optional[str],
        state_key: str,
        action_name: str,
        action_family: str,
        action_instance_id: str,
        objective_outcome,
        source_event: str,
        decision_id: Optional[int] = None,
        transition_id: Optional[int] = None,
        success: Optional[bool] = None,
        scalarization_profile_instance_id: Optional[
            str
        ] = None,
        derived_scalar_utility: Optional[
            float
        ] = None,
    ) -> ObjectiveExperienceRecord:
        """
        Persist one exact ACTUAL objective vector into SQLite COLD history.

        This method is archive-only. It must never update Q, world-model
        statistics, calibration, Evidence, or joint sufficient statistics.
        """
        structured = ObjectiveOutcome.coerce(
            objective_outcome
        )
        record = ObjectiveExperienceRecord(
            experience_id=(
                "objexp-"
                + uuid.uuid4().hex
            ),
            context=context,
            belief_context_id=(
                belief_context_id
            ),
            state_key=state_key,
            action_name=action_name,
            action_family=action_family,
            action_instance_id=(
                action_instance_id
            ),
            objective_outcome=(
                structured.as_dict()
            ),
            source_event=source_event,
            decision_id=decision_id,
            transition_id=transition_id,
            observed_at=(
                self.interaction_clock
            ),
            success=success,
            scalarization_profile_instance_id=(
                scalarization_profile_instance_id
            ),
            derived_scalar_utility=(
                derived_scalar_utility
            ),
        )
        self.epistemic_archive.archive_objective_experience(
            record
        )
        return record

    def objective_experience_history(
        self,
        context: Optional[str] = None,
        action_name: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        state_key: Optional[str] = None,
        action_instance_id: Optional[
            str
        ] = None,
    ) -> List[ObjectiveExperienceRecord]:
        """
        Read exact archived actual objective vectors.

        If context/action are supplied, current explicit identity adapters
        resolve the query. Historical rows themselves are never migrated.
        """
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )

        resolved_state = state_key
        if (
            resolved_state is None
            and context is not None
        ):
            resolved_state = (
                self.state_learning_key(
                    context
                )
            )

        resolved_action = (
            action_instance_id
        )
        if (
            resolved_action is None
            and action_name is not None
        ):
            resolved_action = (
                self.resolve_action_identity(
                    action_name,
                    require_active=False,
                ).instance_id
            )

        return (
            self.epistemic_archive
            .objective_experiences(
                belief_context_id=(
                    resolved_scope
                ),
                state_key=(
                    resolved_state
                ),
                action_instance_id=(
                    resolved_action
                ),
            )
        )

    def replay_objective_experience_utility(
        self,
        context: str,
        action_name: str,
        objective_profile_reference: Optional[
            str
        ] = None,
        belief_context_id: Optional[str] = None,
        include_records: bool = False,
    ) -> Dict:
        """
        Read-only exact replay of archived objective vectors under a profile.

        Per-record missing-component normalization is preserved exactly.
        A record whose observed mask has zero active profile weight is
        UNSCORABLE, never utility zero.
        """
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = (
            self.state_learning_key(
                context
            )
        )
        identity = (
            self.resolve_action_identity(
                action_name,
                require_active=False,
            )
        )
        profile = (
            self.resolve_objective_profile(
                objective_profile_reference
            )
        )
        aggregator = (
            ObjectiveUtilityAggregator(
                profile
            )
        )

        records = (
            self.epistemic_archive
            .objective_experiences(
                belief_context_id=(
                    resolved_scope
                ),
                state_key=state_key,
                action_instance_id=(
                    identity.instance_id
                ),
            )
        )

        utilities = []
        per_record = []
        unscorable = 0

        for record in records:
            try:
                aggregation = (
                    aggregator.aggregate(
                        record.objective_outcome
                    )
                )
            except ValueError as exc:
                if (
                    "total weight 0"
                    not in str(exc)
                ):
                    raise
                unscorable += 1
                if include_records:
                    per_record.append({
                        "experience_id":
                            record.experience_id,
                        "component_mask":
                            record.component_mask,
                        "scorable":
                            False,
                        "reason":
                            "active_profile_weight_zero_for_mask",
                    })
                continue

            utility = float(
                aggregation.scalar_utility
            )
            utilities.append(
                utility
            )
            if include_records:
                per_record.append({
                    "experience_id":
                        record.experience_id,
                    "component_mask":
                        record.component_mask,
                    "scorable":
                        True,
                    "utility":
                        utility,
                    "scalarization_profile_instance_id":
                        record.scalarization_profile_instance_id,
                })

        count = len(
            utilities
        )
        total = len(
            records
        )

        if count <= 0:
            mean = None
            variance = None
            std = None
            minimum = None
            maximum = None
        else:
            mean = sum(
                utilities
            ) / count
            minimum = min(
                utilities
            )
            maximum = max(
                utilities
            )
            if count < 2:
                variance = None
                std = None
            else:
                variance = max(
                    0.0,
                    sum(
                        (
                            utility
                            - mean
                        ) ** 2
                        for utility
                        in utilities
                    )
                    / (
                        count
                        - 1
                    ),
                )
                if variance < 1e-15:
                    variance = 0.0
                std = math.sqrt(
                    variance
                )

        joint = (
            self.joint_objective_model
            .reweighted_distribution(
                state_key,
                identity.instance_id,
                profile,
                belief_context_id=(
                    resolved_scope
                ),
            )
        )

        def delta(
            left,
            right,
        ):
            if (
                left is None
                or right is None
            ):
                return None
            return float(
                left - right
            )

        mean_delta = delta(
            mean,
            joint["mean"],
        )
        variance_delta = delta(
            variance,
            joint["variance"],
        )

        joint_total = int(
            joint["total_count"]
        )
        legacy_unarchived_count = max(
            0,
            joint_total - total,
        )
        archive_surplus_count = max(
            0,
            total - joint_total,
        )

        if (
            total == 0
            and joint_total == 0
        ):
            history_completeness = (
                "empty"
            )
        elif total == joint_total:
            history_completeness = (
                "complete"
            )
        elif total < joint_total:
            history_completeness = (
                "partial_legacy_unarchived"
            )
        else:
            history_completeness = (
                "archive_surplus_or_model_gap"
            )

        exact_comparison_applicable = (
            total == joint_total
        )

        exact_agreement = (
            exact_comparison_applicable
            and count
                == joint["scorable_count"]
            and unscorable
                == joint[
                    "unscorable_count"
                ]
            and (
                mean_delta is None
                or abs(
                    mean_delta
                ) <= 1e-12
            )
            and (
                variance_delta is None
                or abs(
                    variance_delta
                ) <= 1e-12
            )
        )

        result = {
            "source":
                "cold_objective_experience_replay",
            "learning_mutation":
                False,
            "belief_context_id":
                resolved_scope,
            "context":
                context,
            "state_key":
                state_key,
            "action_family":
                identity.family,
            "action_instance_id":
                identity.instance_id,
            "objective_profile_instance_id":
                profile.instance_id,
            "objective_profile_signature":
                profile.signature,
            "total_count":
                total,
            "scorable_count":
                count,
            "unscorable_count":
                unscorable,
            "coverage": (
                count / total
                if total > 0
                else 0.0
            ),
            "mean":
                mean,
            "variance":
                variance,
            "std":
                std,
            "min":
                minimum,
            "max":
                maximum,
            "exact_missingness":
                True,
            "record_level_history": {
                "completeness":
                    history_completeness,
                "legacy_unarchived_count":
                    legacy_unarchived_count,
                "archive_surplus_count":
                    archive_surplus_count,
                "coverage_of_joint_history": (
                    (
                        total
                        / joint_total
                    )
                    if joint_total > 0
                    else (
                        1.0
                        if total == 0
                        else 0.0
                    )
                ),
                "synthetic_records_created":
                    False,
            },
            "joint_reference": {
                "total_count":
                    joint[
                        "total_count"
                    ],
                "scorable_count":
                    joint[
                        "scorable_count"
                    ],
                "unscorable_count":
                    joint[
                        "unscorable_count"
                    ],
                "coverage":
                    joint["coverage"],
                "mean":
                    joint["mean"],
                "variance":
                    joint["variance"],
                "std":
                    joint["std"],
            },
            "joint_agreement": {
                "comparison_applicable":
                    exact_comparison_applicable,
                "mean_delta": (
                    mean_delta
                    if exact_comparison_applicable
                    else None
                ),
                "variance_delta": (
                    variance_delta
                    if exact_comparison_applicable
                    else None
                ),
                "exact_within_tolerance":
                    exact_agreement,
            },
        }

        if include_records:
            result[
                "records"
            ] = per_record

        return result

    def aggregate_objective_outcome(
        self,
        objective_outcome,
        objective_profile_reference: Optional[
            str
        ] = None,
    ) -> ObjectiveAggregation:
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )
        return ObjectiveUtilityAggregator(
            profile
        ).aggregate(
            objective_outcome
        )

    def _resolve_actual_utility(
        self,
        reward: Optional[float],
        objective_outcome=None,
        objective_profile_reference: Optional[
            str
        ] = None,
    ):
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )

        structured = (
            None
            if objective_outcome is None
            else ObjectiveOutcome.coerce(
                objective_outcome
            )
        )
        aggregation = (
            None
            if structured is None
            else self.aggregate_objective_outcome(
                structured,
                objective_profile_reference=(
                    profile.instance_id
                ),
            )
        )

        if reward is None:
            if aggregation is None:
                raise ValueError(
                    "reward atau objective_outcome wajib diberikan"
                )
            resolved = (
                aggregation.scalar_utility
            )
        else:
            resolved = float(reward)
            if not 0.0 <= resolved <= 1.0:
                raise ValueError(
                    "reward harus 0..1"
                )
            if (
                aggregation is not None
                and abs(
                    resolved
                    - aggregation.scalar_utility
                ) > 1e-9
            ):
                raise ValueError(
                    "reward scalar tidak konsisten "
                    "dengan objective_outcome "
                    f"under {profile.instance_id}"
                )

        return (
            resolved,
            structured,
            aggregation,
        )

    def _objective_scalar_state_key_from_canonical(
        self,
        canonical_state_key: str,
        objective_profile_reference: Optional[
            str
        ] = None,
    ) -> str:
        """
        Scalar utility namespace.

        Initial V2.26-compatible profile deliberately keeps the old raw
        canonical-state key. Successor profiles use isolated state namespaces,
        so old scalar Q/world-model/calibration cannot leak into a new
        preference interpretation.
        """
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )

        if (
            profile.instance_id
            == self._objective_compatibility_instance_id
        ):
            return canonical_state_key

        return (
            "@objective["
            f"{profile.instance_id}"
            "]::"
            f"{canonical_state_key}"
        )

    def objective_scalar_state_key(
        self,
        context: str,
        objective_profile_reference: Optional[
            str
        ] = None,
    ) -> str:
        return (
            self._objective_scalar_state_key_from_canonical(
                self.state_learning_key(
                    context
                ),
                objective_profile_reference=(
                    objective_profile_reference
                ),
            )
        )

    def objective_profile_state(
        self,
        objective_profile_reference: Optional[
            str
        ] = None,
    ) -> Dict:
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )
        return {
            "profile_id":
                profile.profile_id,
            "profile_version":
                profile.profile_version,
            "profile_instance_id":
                profile.instance_id,
            "profile_signature":
                profile.signature,
            "weights":
                profile.weights(),
            "valid_from":
                profile.valid_from,
            "valid_until":
                profile.valid_until,
            "note":
                profile.note,
            "compatibility_instance_id":
                self._objective_compatibility_instance_id,
            "components":
                tuple(
                    OBJECTIVE_COMPONENTS
                ),
            "cost_components":
                ("execution_cost",),
            "missing_semantics":
                "renormalize_observed_not_zero",
            "registry":
                self.objective_profile_registry.state(),
        }

    def supersede_objective_profile(
        self,
        *,
        task_progress_weight: float,
        correctness_weight: float,
        execution_cost_weight: float,
        reversibility_weight: float,
        user_acceptance_weight: float,
        observed_at: Optional[int] = None,
        reason: str = "",
    ) -> Dict:
        """
        Explicit preference change.

        Does NOT:
        - create a BeliefContext,
        - rewrite old Q,
        - rewrite old scalar world-model samples,
        - fabricate new objective experience.

        Historical vector objective statistics remain reusable for read-only
        reweighting because those are measurements, not preference judgments.
        """
        resolved_at = (
            self.interaction_clock
            if observed_at is None
            else observed_at
        )
        if (
            resolved_at
            < self.interaction_clock
        ):
            raise ValueError(
                "Objective profile supersession "
                "tidak boleh mundur dari interaction_clock"
            )

        before_context = (
            self.belief_contexts.current_id
        )

        successor = (
            ObjectiveUtilityProfile(
                profile_id=(
                    self.objective_profile.profile_id
                ),
                task_progress_weight=(
                    task_progress_weight
                ),
                correctness_weight=(
                    correctness_weight
                ),
                execution_cost_weight=(
                    execution_cost_weight
                ),
                reversibility_weight=(
                    reversibility_weight
                ),
                user_acceptance_weight=(
                    user_acceptance_weight
                ),
                note=reason,
            )
        )

        (
            previous,
            current,
        ) = (
            self.objective_profile_registry
            .supersede(
                self.objective_profile.instance_id,
                successor,
                observed_at=resolved_at,
            )
        )

        self.objective_profile = current
        self.objective_aggregator = (
            ObjectiveUtilityAggregator(
                current
            )
        )
        self.touch_interaction_time(
            resolved_at
        )

        if (
            self.belief_contexts.current_id
            != before_context
        ):
            raise RuntimeError(
                "Objective profile supersession "
                "tidak boleh mengubah BeliefContext"
            )

        return {
            "operation":
                "supersede_objective_profile",
            "observed_at":
                resolved_at,
            "reason":
                reason,
            "belief_context_id":
                before_context,
            "previous_profile":
                self.objective_profile_state(
                    previous.instance_id
                ),
            "current_profile":
                self.objective_profile_state(
                    current.instance_id
                ),
            "q_transfer":
                "none",
            "scalar_world_model_transfer":
                "none",
            "vector_history":
                "reweightable_read_only",
        }

    def reweighted_objective_estimate(
        self,
        context: str,
        action_name: str,
        objective_profile_reference: Optional[
            str
        ] = None,
        belief_context_id: Optional[
            str
        ] = None,
    ) -> Dict:
        """
        Read-only reinterpretation of ACTUAL objective-vector history.

        No Q/world-model/sample counters are updated. This is preference
        reinterpretation, not experience transfer.
        """
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )
        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        canonical_state = (
            self.state_learning_key(
                context
            )
        )
        identity = (
            self.resolve_action_identity(
                action_name,
                require_active=False,
            )
        )

        component_stats = (
            self.objective_world_model
            .statistics(
                canonical_state,
                identity.instance_id,
                belief_context_id=scope,
            )
        )

        means = {
            component:
                values["mean"]
            for (
                component,
                values,
            ) in component_stats.items()
            if values["count"] > 0
        }
        component_counts = {
            component:
                values["count"]
            for (
                component,
                values,
            ) in component_stats.items()
            if values["count"] > 0
        }

        distribution = (
            self.joint_objective_model
            .reweighted_distribution(
                canonical_state,
                identity.instance_id,
                profile,
                belief_context_id=scope,
            )
        )

        return {
            "belief_context_id":
                scope,
            "context":
                context,
            "state_key":
                canonical_state,
            "action_name":
                action_name,
            "action_instance_id":
                identity.instance_id,
            "objective_profile_instance_id":
                profile.instance_id,
            "objective_profile_signature":
                profile.signature,
            # Marginal means remain useful diagnostics, but are NOT used
            # to construct scalar utility in V2.28.
            "predicted_objectives":
                means,
            "component_counts":
                component_counts,
            "utility":
                distribution[
                    "mean"
                ],
            "variance":
                distribution[
                    "variance"
                ],
            "std":
                distribution[
                    "std"
                ],
            "support":
                distribution[
                    "scorable_count"
                ],
            "total_count":
                distribution[
                    "total_count"
                ],
            "unscorable_count":
                distribution[
                    "unscorable_count"
                ],
            "coverage":
                distribution[
                    "coverage"
                ],
            "mask_count":
                distribution[
                    "mask_count"
                ],
            "scorable_mask_count":
                distribution[
                    "scorable_mask_count"
                ],
            "mask_breakdown":
                distribution[
                    "mask_breakdown"
                ],
            "exact_missingness":
                True,
            "source":
                "actual_joint_vector_history_reweighted",
            "learning_mutation":
                False,
        }


    def objective_world_model_state(
        self,
        context: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = (
            self.state_learning_key(
                context
            )
            if context is not None
            else None
        )
        return {
            "belief_context_id":
                resolved_scope,
            "context":
                context,
            "state_key":
                state_key,
            "profile":
                self.objective_profile_state(),
            "stats":
                self.objective_world_model.state(
                    belief_context_id=(
                        resolved_scope
                    ),
                    context=state_key,
                ),
            "joint_stats":
                self.joint_objective_model.state(
                    belief_context_id=(
                        resolved_scope
                    ),
                    context=state_key,
                ),
            "joint_group_count":
                self.joint_objective_model.group_count(
                    belief_context_id=(
                        resolved_scope
                    ),
                    context=state_key,
                ),
        }


    # -----------------------------------------------------------------
    # V2.25 — STATE IDENTITY / CANONICALIZATION
    # -----------------------------------------------------------------

    def register_state_equivalence(
        self,
        canonical_id: str,
        equivalence_fingerprint: str,
        aliases: Tuple[str, ...] = (),
        note: str = "",
    ) -> StateCanonicalDefinition:
        return self.state_registry.register(
            canonical_id=canonical_id,
            equivalence_fingerprint=equivalence_fingerprint,
            aliases=aliases,
            note=note,
        )

    def register_state_alias(
        self,
        canonical_id: str,
        alias: str,
    ) -> StateCanonicalDefinition:
        return self.state_registry.add_alias(
            canonical_id,
            alias,
        )

    def resolve_state_identity(
        self,
        state_reference: str,
    ) -> ResolvedStateIdentity:
        return self.state_registry.resolve(
            state_reference
        )

    def state_learning_key(
        self,
        state_reference: str,
    ) -> str:
        return self.resolve_state_identity(
            state_reference
        ).canonical_id

    def state_registry_state(
        self,
        canonical_id: Optional[str] = None,
    ) -> Dict:
        return self.state_registry.state(
            canonical_id
        )

    @staticmethod
    def canonicalize_structured_state(
        mapping: Dict,
        include_fields: Optional[Tuple[str, ...]] = None,
        exclude_fields: Tuple[str, ...] = (),
    ) -> str:
        return StateIdentityRegistry.canonical_mapping(
            mapping,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
        )

    def state_model_scope(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        return (
            scope,
            self.state_learning_key(context),
        )

    # -----------------------------------------------------------------
    # V2.22 — ACTION IDENTITY / VERSIONING
    # -----------------------------------------------------------------

    def register_action(
        self,
        action_id: str,
        implementation_fingerprint: str,
        valid_from: Optional[int] = None,
        valid_until: Optional[int] = None,
        action_version: Optional[int] = None,
        note: str = "",
    ) -> ActionDefinition:
        resolved_from = (
            self.interaction_clock
            if valid_from is None
            else valid_from
        )
        return self.action_registry.register(
            ActionDefinition(
                action_id=action_id,
                implementation_fingerprint=(
                    implementation_fingerprint
                ),
                valid_from=resolved_from,
                valid_until=valid_until,
                action_version=action_version,
                note=note,
            )
        )

    def supersede_action(
        self,
        action_id: str,
        implementation_fingerprint: str,
        observed_at: Optional[int] = None,
        note: str = "",
    ) -> Dict:
        resolved_at = (
            self.interaction_clock
            if observed_at is None
            else observed_at
        )
        if resolved_at < self.interaction_clock:
            raise ValueError(
                "Action supersession tidak boleh mundur dari "
                "interaction_clock"
            )
        self.touch_interaction_time(
            resolved_at
        )
        return self.action_registry.supersede(
            action_id=action_id,
            implementation_fingerprint=(
                implementation_fingerprint
            ),
            observed_at=resolved_at,
            note=note,
        )

    def action_versions(
        self,
        action_id: str,
    ) -> List[ActionDefinition]:
        return self.action_registry.all_versions(
            action_id
        )

    def action_registry_state(
        self,
        action_id: Optional[str] = None,
    ) -> Dict:
        state = self.action_registry.state(
            action_id
        )
        state["interaction_clock"] = (
            self.interaction_clock
        )
        return state

    def resolve_action_identity(
        self,
        action_reference: str,
        as_of: Optional[int] = None,
        require_active: bool = False,
    ) -> ResolvedActionIdentity:
        resolved_time = (
            self.interaction_clock
            if as_of is None
            else as_of
        )
        definition = self.action_registry.resolve(
            action_reference,
            as_of=resolved_time,
            require_active=require_active,
        )
        if definition is None:
            # Legacy compatibility path: the raw string is the learning key.
            return ResolvedActionIdentity(
                reference=action_reference,
                family=action_reference,
                instance_id=action_reference,
                registered=False,
            )

        return ResolvedActionIdentity(
            reference=action_reference,
            family=definition.action_id,
            instance_id=definition.instance_id,
            registered=True,
            action_version=(
                definition.action_version
            ),
            implementation_fingerprint=(
                definition.implementation_fingerprint
            ),
        )

    def action_learning_key(
        self,
        action_reference: str,
        as_of: Optional[int] = None,
        require_active: bool = False,
    ) -> str:
        """
        Public adapter helper for standalone V3 ensemble/conformal tooling.

        Feed this key to external model components when an action is
        registered, so their own string-keyed statistics inherit V2.22
        version isolation.
        """
        return self.resolve_action_identity(
            action_reference,
            as_of=as_of,
            require_active=require_active,
        ).instance_id

    def action_model_scope(
        self,
        context: str,
        action_reference: str,
        belief_context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[Optional[str], str, str]:
        """
        Canonical scope for standalone V3 ensemble/conformal components.
        """
        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        identity = self.resolve_action_identity(
            action_reference,
            as_of=as_of,
            require_active=False,
        )
        return (
            scope,
            self.objective_scalar_state_key(
                context
            ),
            identity.instance_id,
        )

    def bind_action_risk_estimate(
        self,
        estimate: ActionRiskEstimate,
        context: Optional[str] = None,
    ) -> ActionRiskEstimate:
        """
        Adapter convenience: attach the current immutable action instance to
        an ActionRiskEstimate without altering its risk values.
        """
        identity = self.resolve_action_identity(
            estimate.action,
            require_active=True,
        )
        return replace(
            estimate,
            action_instance_id=(
                identity.instance_id
                if identity.registered
                else estimate.action_instance_id
            ),
            state_key=(
                self.state_learning_key(context)
                if context is not None
                else estimate.state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
        )

    def _resolve_action_candidates(
        self,
        candidates: List[str],
    ) -> Dict[str, ResolvedActionIdentity]:
        references = sorted(
            set(candidates)
        )
        if not references:
            raise ValueError(
                "Candidates tidak boleh kosong"
            )

        identities = {
            reference:
                self.resolve_action_identity(
                    reference,
                    require_active=True,
                )
            for reference in references
        }

        instance_ids = [
            identity.instance_id
            for identity in identities.values()
        ]
        if len(instance_ids) != len(
            set(instance_ids)
        ):
            raise ActionVersionConflict(
                "Dua candidate reference resolve ke action instance yang sama"
            )

        return identities

    @staticmethod
    def _to_instance_mapping(
        values: Optional[Dict[str, float]],
        identities: Dict[
            str,
            ResolvedActionIdentity,
        ],
    ) -> Optional[Dict[str, float]]:
        if values is None:
            return None

        unknown = set(values) - set(
            identities
        )
        if unknown:
            raise ValueError(
                "Mapping memiliki action yang bukan kandidat: "
                f"{sorted(unknown)}"
            )

        return {
            identities[reference].instance_id:
                value
            for reference, value
            in values.items()
        }

    @staticmethod
    def _translate_instance_dict(
        values: Dict[str, float],
        identities: Dict[
            str,
            ResolvedActionIdentity,
        ],
    ) -> Dict[str, float]:
        reverse = {
            identity.instance_id:
                reference
            for reference, identity
            in identities.items()
        }
        return {
            reverse.get(key, key): value
            for key, value in values.items()
        }

    def _validate_action_estimate_identity(
        self,
        reference: str,
        estimate: ActionRiskEstimate,
        context: Optional[str] = None,
    ) -> ResolvedActionIdentity:
        identity = self.resolve_action_identity(
            reference,
            require_active=True,
        )
        if (
            context is not None
            and estimate.state_key is not None
            and estimate.state_key
                != self.state_learning_key(context)
        ):
            raise StateIdentityConflict(
                "Stale/mismatched ActionRiskEstimate state: "
                f"{estimate.state_key} != "
                f"{self.state_learning_key(context)}"
            )
        estimate_profile = getattr(
            estimate,
            "objective_profile_instance_id",
            None,
        )
        current_profile = (
            self.objective_profile.instance_id
        )

        if estimate_profile is None:
            if (
                current_profile
                != self._objective_compatibility_instance_id
            ):
                raise ObjectiveProfileVersionConflict(
                    "ActionRiskEstimate tanpa objective_profile_instance_id "
                    "ambigu setelah preference supersession"
                )
        elif (
            estimate_profile
            != current_profile
        ):
            raise ObjectiveProfileVersionConflict(
                "Stale/mismatched ActionRiskEstimate objective profile: "
                f"{estimate_profile} != {current_profile}"
            )

        if estimate.action != reference:
            raise ValueError(
                "ActionRiskEstimate.action harus sama dengan candidate reference"
            )

        if identity.registered:
            if estimate.action_instance_id is None:
                raise ActionVersionConflict(
                    f"ActionRiskEstimate untuk registered action "
                    f"'{reference}' wajib membawa action_instance_id="
                    f"{identity.instance_id}"
                )
            if (
                estimate.action_instance_id
                != identity.instance_id
            ):
                raise ActionVersionConflict(
                    "Stale/mismatched ActionRiskEstimate: "
                    f"{estimate.action_instance_id} != "
                    f"{identity.instance_id}"
                )
        elif (
            estimate.action_instance_id is not None
            and estimate.action_instance_id
                != identity.instance_id
        ):
            raise ActionVersionConflict(
                "Legacy action estimate memiliki instance id yang tidak cocok"
            )

        return identity


    def maintain_epistemic_archive(
        self,
    ) -> Dict[str, int]:
        policy = self.epistemic_archive_policy
        if not policy.enabled:
            return {
                "evidence": 0,
                "episodes": 0,
            }

        archived = {
            "evidence": 0,
            "episodes": 0,
        }

        evidence_trigger = (
            policy.evidence_hot_limit
            + policy.archive_batch
        )
        if len(
            self.evidence_pool
        ) > evidence_trigger:
            target = (
                len(self.evidence_pool)
                - policy.evidence_hot_limit
            )
            batch = list(
                self.evidence_pool[:target]
            )
            written = (
                self.epistemic_archive
                .archive_evidence_batch(
                    batch
                )
            )
            if written != len(batch):
                raise RuntimeError(
                    "Cold archive evidence incomplete"
                )
            del self.evidence_pool[:target]
            archived["evidence"] = written

        episode_trigger = (
            policy.episode_hot_limit
            + policy.archive_batch
        )
        if len(
            self.memory._episodes
        ) > episode_trigger:
            target = (
                len(self.memory._episodes)
                - policy.episode_hot_limit
            )
            batch = list(
                self.memory._episodes[:target]
            )
            written = (
                self.epistemic_archive
                .archive_episode_batch(
                    batch
                )
            )
            if written != len(batch):
                raise RuntimeError(
                    "Cold archive episodes incomplete"
                )
            del self.memory._episodes[:target]
            archived["episodes"] = written

        return archived

    def epistemic_archive_state(
        self,
    ) -> Dict:
        cold = self.epistemic_archive.state()
        return {
            "enabled":
                self.epistemic_archive_policy.enabled,
            "hot": {
                "evidence":
                    len(self.evidence_pool),
                "episodes":
                    len(self.memory._episodes),
                "groundings":
                    len(self.grounding_store.records),
                "justifications":
                    len(self.justifications),
            },
            "cold": {
                "evidence":
                    cold["evidence_records"],
                "episodes":
                    cold["episode_records"],
                "objective_experiences":
                    cold.get(
                        "objective_experience_records",
                        0,
                    ),
            },
            "total": {
                "evidence": (
                    len(self.evidence_pool)
                    + cold["evidence_records"]
                ),
                "episodes": (
                    len(self.memory._episodes)
                    + cold["episode_records"]
                ),
                "objective_experiences":
                    cold.get(
                        "objective_experience_records",
                        0,
                    ),
            },
            "archive_file_bytes":
                cold["file_bytes"],
            "observation_group_records":
                cold.get(
                    "observation_group_records",
                    0,
                ),
            "archive_path":
                cold["path"],
            "protected_in_ram": (
                "GroundingStore + Justifications"
            ),
        }

    def _evidence_for_claim_exact(
        self,
        claim_id: str,
    ) -> List[Evidence]:
        archived = (
            self.epistemic_archive
            .evidence_for_claim(
                claim_id
            )
        )
        hot = [
            evidence
            for evidence in self.evidence_pool
            if evidence.claim_id == claim_id
        ]
        return archived + hot

    def all_evidence(
        self,
    ) -> List[Evidence]:
        return (
            self.epistemic_archive
            .all_evidence()
            + list(self.evidence_pool)
        )

    def evidence_query_state(
        self,
    ) -> Dict:
        state = (
            self.evidence_query_engine
            .state()
        )
        state["evidence_revision"] = (
            self._evidence_revision
        )
        return state

    def source_observation_reliability(
        self,
        source: str,
    ) -> float:
        profile = self.sources.get(source)
        return (
            1.0
            if profile is None
            else profile.observation_reliability
        )

    def observation_reliability_state(
        self,
    ) -> Dict:
        return {
            "sources": {
                name: {
                    "factual_reliability": profile.reliability,
                    "observation_reliability": profile.observation_reliability,
                    "consistent_retry_groups": (
                        profile.observation_consistent_groups
                    ),
                    "conflicting_retry_groups": (
                        profile.observation_conflicting_groups
                    ),
                }
                for name, profile
                in sorted(self.sources.items())
            },
            "retry_group_records": (
                self.epistemic_archive
                .observation_group_count()
            ),
        }

    def _evidence_for_retry_group_exact(
        self,
        source: str,
        retry_group_id: str,
    ) -> List[Evidence]:
        archived = (
            self.epistemic_archive
            .evidence_for_retry_group(
                source,
                retry_group_id,
            )
        )
        hot = [
            evidence
            for evidence in self.evidence_pool
            if (
                evidence.source == source
                and evidence.retry_group_id
                    == retry_group_id
            )
        ]
        return archived + hot

    def _update_observation_reliability(
        self,
        evidence: Evidence,
    ) -> Dict:
        retry_group_id = evidence.retry_group_id
        profile = self.sources.get(
            evidence.source
        )
        if (
            retry_group_id is None
            or profile is None
        ):
            return {
                "updated": False,
                "status": None,
                "previous_status": None,
            }

        records = (
            self._evidence_for_retry_group_exact(
                evidence.source,
                retry_group_id,
            )
        )

        # Duplicate storage copies with identical evidence_id do not turn one
        # observation into multiple retries.
        unique = {}
        for item in records:
            unique.setdefault(
                item.evidence_id,
                item,
            )
        records = list(unique.values())

        polarities = {
            item.polarity
            for item in records
        }
        if len(records) < 2:
            status = "pending"
        elif len(polarities) == 1:
            status = "consistent"
        else:
            status = "conflicting"

        previous = (
            self.epistemic_archive
            .observation_group_status(
                evidence.source,
                retry_group_id,
            )
        )

        if previous != status:
            profile.record_observation_group_transition(
                previous,
                status,
            )
            self.epistemic_archive.set_observation_group_status(
                evidence.source,
                retry_group_id,
                status,
            )

        return {
            "updated": previous != status,
            "status": status,
            "previous_status": previous,
            "observation_reliability": (
                profile.observation_reliability
            ),
            "unique_retry_records": len(records),
        }

    def maintain_memory(
        self,
        memory_names: Optional[
            Tuple[str, ...]
        ] = None,
    ) -> Dict[str, int]:
        return self.memory_lifecycle.maintain(
            memory_names
        )

    def memory_lifecycle_state(self) -> Dict:
        return self.memory_lifecycle.state()

    def _repair_runtime_links(self):
        """
        Repair explicit owner/source back-references after restart.

        Pickle normally preserves these graph links, but repairing them makes
        the runtime invariant explicit and protects against future manager
        refactors.
        """
        self.evidence_aggregator.sources = self.sources
        self.evidence_aggregator.lineage = (
            SourceLineage(
                self.sources
            )
        )

        self.rule_validator.domain = self.domain

        if not hasattr(
            self.rule_validator,
            "versions",
        ):
            self.rule_validator.versions = {}
            for rid, rule in (
                self.rule_validator.rules.items()
            ):
                versioned = (
                    rule
                    if rule.rule_version is not None
                    else replace(
                        rule,
                        rule_version=1,
                    )
                )
                self.rule_validator.versions[
                    rid
                ] = [versioned]
                self.rule_validator.rules[
                    rid
                ] = versioned

        self.truth_evaluator.rules = (
            self.rule_validator
        )

        self.tms.agent = self
        self.memory_lifecycle.agent = self

        if hasattr(
            self,
            "epistemic_archive",
        ):
            self.memory.archive_manager = (
                self.epistemic_archive
            )

        if not hasattr(
            self,
            "_evidence_revision",
        ):
            self._evidence_revision = 0

        if not hasattr(
            self,
            "evidence_query_engine",
        ):
            self.evidence_query_engine = (
                ExactEvidenceQueryEngine(
                    self
                )
            )
        else:
            self.evidence_query_engine.agent = self
            self.evidence_query_engine.clear()

        if not hasattr(
            self,
            "_active_prediction_pins",
        ):
            self._active_prediction_pins = set()

        # V2.32 compatibility backfill: previous checkpoints/portable states
        # contain no structural-pattern history. Add an empty store; never
        # synthesize pattern observations from unrelated historical records.
        if not hasattr(
            self,
            "structural_patterns",
        ):
            self.structural_patterns = StructuralPatternStore()

        # V2.33 compatibility backfill. Frozen predecessors contain no
        # object-centric spatial scene history. Add an EMPTY scene store and
        # never reinterpret old planner/grid state as V2.33 spatial scenes.
        if not hasattr(
            self,
            "spatial_scenes",
        ):
            self.spatial_scenes = SpatialSceneStore()

        # V2.37 compatibility backfill: predecessor states contain no execution
        # tickets or actual manipulation feedback. Add an EMPTY bounded store;
        # never synthesize dispatch/observation history from plans or scenes.
        if not hasattr(
            self,
            "spatial_execution",
        ):
            self.spatial_execution = SpatialExecutionStore()

        # V2.38 compatibility backfill: predecessor states have no replanning
        # journal. Add an EMPTY bounded store; never reinterpret old plans,
        # feedback, or deviations as historical replan attempts.
        if not hasattr(
            self,
            "spatial_replanning",
        ):
            self.spatial_replanning = SpatialReplanningStore()

        # V2.39 compatibility backfill: predecessor states have no recovery
        # decision journal. Add EMPTY state; never reinterpret old feedback or
        # replan records as historical recovery-policy decisions.
        if not hasattr(
            self,
            "spatial_recovery",
        ):
            self.spatial_recovery = SpatialRecoveryStore()

        # V2.40 compatibility backfill: predecessor states may contain CLOSED
        # execution feedback, but no reliability history. Add EMPTY state and
        # never retroactively synthesize samples from old feedback.
        if not hasattr(
            self,
            "spatial_reliability",
        ):
            self.spatial_reliability = SpatialManipulationReliabilityStore()

        if not hasattr(
            self,
            "interaction_clock",
        ):
            self.interaction_clock = (
                self.belief_contexts.now
            )

        # The actual-interaction clock may legitimately be ahead of older
        # belief-only timestamps, but never behind them.
        if (
            self.interaction_clock
            < self.belief_contexts.now
        ):
            self.interaction_clock = (
                self.belief_contexts.now
            )



        # V2.23 robust belief-revision compatibility backfill.
        if not hasattr(
            self,
            "belief_revision_policy",
        ) or not hasattr(
            self.belief_revision_policy,
            "assess",
        ):
            self.belief_revision_policy = (
                ContextualBeliefRevisionPolicy()
            )

        if not hasattr(
            self,
            "belief_shift_memory",
        ):
            self.belief_shift_memory = (
                BeliefShiftDecisionMemory(
                    limit=2048
                )
            )

        if not hasattr(
            self,
            "_belief_shift_decision_counter",
        ):
            self._belief_shift_decision_counter = (
                self.belief_shift_memory.total_seen
            )

        # V2.27 objective-profile identity/versioning backfill.
        if not hasattr(
            self,
            "objective_profile",
        ):
            self.objective_profile = (
                ObjectiveUtilityProfile()
            )

        if not hasattr(
            self,
            "objective_profile_registry",
        ):
            self.objective_profile_registry = (
                ObjectiveProfileRegistry()
            )
            self.objective_profile = (
                self.objective_profile_registry
                .register(
                    self.objective_profile,
                    activated_at=0,
                )
            )

        if not hasattr(
            self,
            "_objective_compatibility_instance_id",
        ):
            versions = (
                self.objective_profile_registry
                .all_versions(
                    self.objective_profile.profile_id
                )
            )
            self._objective_compatibility_instance_id = (
                versions[0].instance_id
                if versions
                else self.objective_profile.instance_id
            )

        # Ensure current pointer resolves to the active registry instance.
        try:
            self.objective_profile = (
                self.objective_profile_registry
                .active(
                    self.objective_profile.profile_id
                )
            )
        except KeyError:
            self.objective_profile = (
                self.objective_profile_registry
                .register(
                    self.objective_profile,
                    activated_at=(
                        self.interaction_clock
                    ),
                )
            )

        if not hasattr(
            self,
            "objective_aggregator",
        ):
            self.objective_aggregator = (
                ObjectiveUtilityAggregator(
                    self.objective_profile
                )
            )
        else:
            self.objective_aggregator.profile = (
                self.objective_profile
            )

        if not hasattr(
            self,
            "objective_world_model",
        ):
            self.objective_world_model = (
                ContextScopedObjectiveModel()
            )

        if not hasattr(
            self,
            "joint_objective_model",
        ):
            self.joint_objective_model = (
                ContextScopedJointObjectiveModel()
            )

        if not hasattr(
            self,
            "success_constraint_model",
        ):
            self.success_constraint_model = (
                ContextScopedSuccessConstraintModel()
            )

        # V2.22 compatibility backfill.
        if not hasattr(
            self,
            "action_registry",
        ):
            self.action_registry = (
                ActionRegistry()
            )

        # V2.25 state-identity compatibility backfill.
        if not hasattr(
            self,
            "state_registry",
        ):
            self.state_registry = (
                StateIdentityRegistry()
            )

        if not hasattr(
            self.world_model_reliability,
            "_action_profiles",
        ):
            self.world_model_reliability._action_profiles = {}

        for summary in (
            self.memory_lifecycle.summaries.values()
        ):
            if not hasattr(
                summary,
                "by_action_instance",
            ):
                summary.by_action_instance = {}
            if not hasattr(
                summary,
                "objective_profile_instance_counts",
            ):
                summary.objective_profile_instance_counts = {}

        for record in (
            self.decision_memory._records
        ):
            if not hasattr(
                record,
                "candidate_action_instances",
            ):
                record.candidate_action_instances = {
                    action: action
                    for action in record.candidates
                }
            if not hasattr(
                record,
                "selected_action_family",
            ):
                record.selected_action_family = (
                    record.selected_action
                )
            if not hasattr(
                record,
                "selected_action_instance_id",
            ):
                record.selected_action_instance_id = (
                    record.selected_action
                )

        for record in (
            self.transition_memory._records
        ):
            if not hasattr(
                record,
                "action_family",
            ):
                record.action_family = record.action
            if not hasattr(
                record,
                "action_instance_id",
            ):
                record.action_instance_id = (
                    record.action
                )
            if not hasattr(
                record,
                "next_action_instance_ids",
            ):
                record.next_action_instance_ids = (
                    tuple(record.next_actions)
                )

        for record in (
            self.decision_memory._records
        ):
            if not hasattr(
                record,
                "state_key",
            ):
                record.state_key = (
                    record.context
                )
            if not hasattr(
                record,
                "objective_profile_instance_id",
            ):
                record.objective_profile_instance_id = (
                    self._objective_compatibility_instance_id
                )
            if not hasattr(
                record,
                "scalar_state_key",
            ):
                record.scalar_state_key = (
                    self._objective_scalar_state_key_from_canonical(
                        record.state_key,
                        objective_profile_reference=(
                            record.objective_profile_instance_id
                        ),
                    )
                )

        for record in (
            self.transition_memory._records
        ):
            if not hasattr(
                record,
                "state_key",
            ):
                record.state_key = (
                    record.context
                )
            if not hasattr(
                record,
                "next_state_key",
            ):
                record.next_state_key = (
                    record.next_context
                )
            if not hasattr(
                record,
                "objective_profile_instance_id",
            ):
                record.objective_profile_instance_id = (
                    self._objective_compatibility_instance_id
                )
            if not hasattr(
                record,
                "scalar_state_key",
            ):
                record.scalar_state_key = (
                    self._objective_scalar_state_key_from_canonical(
                        record.state_key,
                        objective_profile_reference=(
                            record.objective_profile_instance_id
                        ),
                    )
                )
            if not hasattr(
                record,
                "next_scalar_state_key",
            ):
                record.next_scalar_state_key = (
                    None
                    if record.next_state_key is None
                    else self._objective_scalar_state_key_from_canonical(
                        record.next_state_key,
                        objective_profile_reference=(
                            record.objective_profile_instance_id
                        ),
                    )
                )

        for record in (
            self.meta_risk_memory._records
        ):
            if not hasattr(
                record,
                "state_key",
            ):
                record.state_key = (
                    record.context
                )
            if not hasattr(
                record,
                "objective_profile_instance_id",
            ):
                record.objective_profile_instance_id = (
                    self._objective_compatibility_instance_id
                )
            if not hasattr(
                record,
                "scalar_state_key",
            ):
                record.scalar_state_key = (
                    self._objective_scalar_state_key_from_canonical(
                        record.state_key,
                        objective_profile_reference=(
                            record.objective_profile_instance_id
                        ),
                    )
                )

        for record in (
            self.prediction_memory._records
        ):
            if not hasattr(
                record,
                "action_instance_id",
            ):
                record.action_instance_id = None
            if not hasattr(
                record,
                "state_key",
            ):
                record.state_key = (
                    record.context
                )
            if not hasattr(
                record,
                "objective_profile_instance_id",
            ):
                record.objective_profile_instance_id = (
                    self._objective_compatibility_instance_id
                )
            if not hasattr(
                record,
                "scalar_state_key",
            ):
                record.scalar_state_key = (
                    self._objective_scalar_state_key_from_canonical(
                        record.state_key,
                        objective_profile_reference=(
                            record.objective_profile_instance_id
                        ),
                    )
                )
            if not hasattr(
                record,
                "reweighted_objective_utility",
            ):
                record.reweighted_objective_utility = None
            if not hasattr(
                record,
                "reweighted_objective_support",
            ):
                record.reweighted_objective_support = 0

        for record in (
            self.prediction_error_memory._records
        ):
            if not hasattr(
                record,
                "action_instance_id",
            ):
                record.action_instance_id = None
            if not hasattr(
                record,
                "state_key",
            ):
                record.state_key = (
                    record.context
                )
            if not hasattr(
                record,
                "objective_profile_instance_id",
            ):
                record.objective_profile_instance_id = (
                    self._objective_compatibility_instance_id
                )
            if not hasattr(
                record,
                "scalar_state_key",
            ):
                record.scalar_state_key = (
                    self._objective_scalar_state_key_from_canonical(
                        record.state_key,
                        objective_profile_reference=(
                            record.objective_profile_instance_id
                        ),
                    )
                )

        if hasattr(
            self,
            "trajectory_decision_memory",
        ):
            for record in (
                self.trajectory_decision_memory._records
            ):
                if not hasattr(
                    record,
                    "objective_profile_instance_id",
                ):
                    record.objective_profile_instance_id = (
                        self._objective_compatibility_instance_id
                    )
                if not hasattr(
                    record,
                    "scalar_state_key",
                ):
                    canonical = (
                        record.state_key
                        if record.state_key is not None
                        else record.context
                    )
                    record.scalar_state_key = (
                        self._objective_scalar_state_key_from_canonical(
                            canonical,
                            objective_profile_reference=(
                                record.objective_profile_instance_id
                            ),
                        )
                    )

        for record in self.world_decision_history:
            if not hasattr(record, "state_key"):
                record.state_key = record.context

        if hasattr(
            self,
            "counterfactual_memory",
        ):
            for record in self.counterfactual_memory._records:
                if not hasattr(record, "state_key"):
                    record.state_key = record.context

        if hasattr(
            self,
            "trajectory_decision_memory",
        ):
            for record in (
                self.trajectory_decision_memory._records
            ):
                if not hasattr(
                    record,
                    "candidate_trajectory_action_instances",
                ):
                    record.candidate_trajectory_action_instances = {
                        trajectory_id:
                            tuple(actions)
                        for trajectory_id, actions
                        in record.candidate_trajectories.items()
                    }
                if not hasattr(
                    record,
                    "selected_action_instance_ids",
                ):
                    record.selected_action_instance_ids = (
                        tuple(
                            record.selected_actions
                        )
                    )
                if not hasattr(
                    record,
                    "selected_first_action_instance_id",
                ):
                    record.selected_first_action_instance_id = (
                        record.selected_first_action
                    )
                if not hasattr(record, "state_key"):
                    record.state_key = record.context

        # Integration backfill: old V2.21 checkpoints remain loadable because
        # CORE_VERSION/schema stay canonical. Only missing V3 optional state is added.
        if not hasattr(self, "trajectory_decision_memory"):
            self.trajectory_decision_memory = TrajectoryDecisionMemory()
        if not hasattr(self, "_trajectory_decision_counter"):
            self._trajectory_decision_counter = 0

        policy = self.uncertainty_decision_policy
        if not hasattr(policy, "safety_gate"):
            policy.safety_gate = ChanceConstrainedSafetyGate()
        if not hasattr(policy, "meta_policy"):
            policy.meta_policy = MetaRiskPolicy()
        self.chance_safety_gate = policy.safety_gate
        self.meta_risk_policy = policy.meta_policy
        if not hasattr(self, "trajectory_chance_safety_gate"):
            self.trajectory_chance_safety_gate = (
                TrajectoryChanceConstrainedSafetyGate(
                    max_trajectory_failure_probability=0.25,
                    action_gate=self.chance_safety_gate,
                )
            )
        else:
            self.trajectory_chance_safety_gate.action_gate = (
                self.chance_safety_gate
            )
        if not hasattr(self, "trajectory_risk_policy"):
            self.trajectory_risk_policy = TrajectoryRiskPolicy(
                safety_gate=self.trajectory_chance_safety_gate,
            )
        else:
            self.trajectory_risk_policy.safety_gate = (
                self.trajectory_chance_safety_gate
            )

        # V2.29 compatibility backfill for trusted-local V2.28/M7 checkpoints.
        if not hasattr(self, "preference_aware_risk_policy"):
            self.preference_aware_risk_policy = PreferenceAwareRiskPolicy(
                safety_gate=self.chance_safety_gate,
            )
        else:
            self.preference_aware_risk_policy.safety_gate = self.chance_safety_gate

        if not hasattr(self, "preference_aware_trajectory_risk_policy"):
            self.preference_aware_trajectory_risk_policy = (
                PreferenceAwareTrajectoryRiskPolicy(
                    safety_gate=self.chance_safety_gate,
                )
            )
        else:
            self.preference_aware_trajectory_risk_policy.safety_gate = (
                self.chance_safety_gate
            )

        return self
    def save_checkpoint(self, path) -> Dict:
        """
        Save an exact trusted-local restart checkpoint atomically.
        """
        return AgentPersistenceManager.save(
            self,
            path,
        )

    @classmethod
    def load_checkpoint(cls, path):
        """
        Restore an exact trusted-local restart checkpoint.

        Never load checkpoints from untrusted sources.
        """
        agent, metadata = (
            AgentPersistenceManager.load(
                path
            )
        )

        if not isinstance(
            agent,
            cls,
        ):
            raise AgentPersistenceError(
                "Checkpoint agent class tidak cocok"
            )

        return agent

    @staticmethod
    def inspect_checkpoint(path) -> Dict:
        return AgentPersistenceManager.inspect(
            path
        )

    def save_portable_state(self, path) -> Dict:
        """Save one language-neutral SQLite cognitive-state database."""
        return PortableCognitiveStateManager.save(
            self,
            path,
        )

    @classmethod
    def load_portable_state(cls, path):
        """Restore from the language-neutral SQLite state format."""
        agent, metadata = (
            PortableCognitiveStateManager.load(
                path
            )
        )
        if not isinstance(agent, cls):
            raise PortableStateTypeError(
                "Portable state agent class tidak cocok"
            )
        return agent

    @staticmethod
    def inspect_portable_state(path) -> Dict:
        return PortableCognitiveStateManager.inspect(
            path
        )

    def persistence_state(self) -> Dict:
        return {
            "core_version": CORE_VERSION,
            "schema_version": (
                PERSISTENCE_SCHEMA_VERSION
            ),
            "portable_state_schema_version": (
                PORTABLE_STATE_SCHEMA_VERSION
            ),
            "portable_state_format": "sqlite3",
            "structural_patterns": (
                self.structural_pattern_state()
            ),
            "spatial_state": (
                self.spatial_state()
            ),
            "domain": self.domain.name,
            "belief_context_id": (
                self.belief_contexts.current_id
            ),
            "belief_time": (
                self.belief_contexts.now
            ),
            "interaction_clock": (
                self.interaction_clock
            ),
            "decision_counter": (
                self._decision_counter
            ),
            "transition_counter": (
                self._transition_counter
            ),
            "prediction_counter": (
                self._outcome_prediction_counter
            ),
            "prediction_error_counter": (
                self._prediction_error_counter
            ),
            "meta_decision_counter": (
                self._meta_decision_counter
            ),
            "active_prediction_pins": (
                sorted(
                    self._active_prediction_pins
                )
            ),
            "memory_lifecycle": (
                self.memory_lifecycle_state()
            ),
            "epistemic_archive": (
                self.epistemic_archive_state()
            ),
            "evidence_query_engine": (
                self.evidence_query_state()
            ),
            "rule_registry": (
                self.rule_registry_state()
            ),
            "observation_reliability": (
                self.observation_reliability_state()
            ),
            "action_registry": (
                self.action_registry_state()
            ),
            "state_registry": (
                self.state_registry_state()
            ),
            "belief_revision": (
                self.belief_revision_state()
            ),
            "objective_profile": (
                self.objective_profile_state()
            ),
            "objective_profile_registry": (
                self.objective_profile_registry.state()
            ),
            "objective_compatibility_instance_id": (
                self._objective_compatibility_instance_id
            ),
            "objective_model_entries": len(
                self.objective_world_model._stats
            ),
            "joint_objective_groups": len(
                self.joint_objective_model._groups
            ),
            "success_constraint_entries": len(
                self.success_constraint_model._stats
            ),
        }

    def add_grounded(
        self,
        claim: str,
        origin: str = "manual",
        note: str = "",
    ):
        """
        Backward-compatible GLOBAL grounding.

        Global facts remain in self.grounded for existing TMS/tests and are
        also registered in GroundingStore.
        """
        self.grounded.add(claim)
        self.grounded_provenance[claim] = {
            "origin": origin,
            "note": note,
            "context_id": None,
            "valid_from": None,
            "valid_until": None,
        }
        self.grounding_store.add(
            GroundedFact(
                claim_id=claim,
                origin=origin,
                note=note,
            )
        )

    def add_contextual_grounded(
        self,
        claim: str,
        origin: str = "manual",
        note: str = "",
        context_id: Optional[str] = None,
        valid_from: Optional[int] = None,
        valid_until: Optional[int] = None,
        global_scope: bool = False,
    ) -> GroundedFact:
        """
        Add a grounding that is scoped by belief context/time.

        Unlike add_grounded(), scoped facts are NOT copied into the legacy
        global self.grounded set.
        """
        resolved_context = (
            None
            if global_scope
            else (
                context_id
                if context_id is not None
                else self.belief_contexts.current_id
            )
        )

        fact = GroundedFact(
            claim_id=claim,
            origin=origin,
            note=note,
            context_id=resolved_context,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        self.grounding_store.add(fact)
        return fact

    def active_grounded(
        self,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Set[str]:
        context_id, as_of = self._resolve_belief_scope(
            context_id=context_id,
            as_of=as_of,
        )

        # Direct legacy mutations of self.grounded remain globally visible.
        active = set(self.grounded)
        active.update(
            self.grounding_store.active_claims(
                context_id=context_id,
                as_of=as_of,
            )
        )
        return active

    def register_source(self, profile: SourceProfile):
        self.sources[profile.name] = profile

    def register_rule(
        self,
        rule: Rule,
    ) -> Rule:
        return self.rule_validator.register(
            rule
        )

    def add_contextual_rule(
        self,
        rule_id: str,
        premises: tuple,
        conclusion: str,
        context_id: Optional[str] = None,
        valid_from: Optional[int] = None,
        valid_until: Optional[int] = None,
        global_scope: bool = False,
        add_justification: bool = True,
    ) -> Rule:
        resolved_context = (
            None
            if global_scope
            else (
                context_id
                if context_id is not None
                else self.belief_contexts.current_id
            )
        )

        rule = Rule(
            rule_id=rule_id,
            domain=self.domain.name,
            premises=premises,
            conclusion=conclusion,
            context_id=resolved_context,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        registered_rule = self.register_rule(
            rule
        )

        if add_justification:
            justification = Justification(
                conclusion=registered_rule.conclusion,
                premises=registered_rule.premises,
                rule_id=registered_rule.rule_id,
                rule_version=registered_rule.rule_version,
                context_id=registered_rule.context_id,
                valid_from=registered_rule.valid_from,
                valid_until=registered_rule.valid_until,
            )
            if justification not in self.justifications:
                self.justifications.append(
                    justification
                )

        return registered_rule


    def rule_versions(
        self,
        rule_id: str,
    ) -> List[Rule]:
        return self.rule_validator.all_versions(
            rule_id
        )

    def rule_registry_state(
        self,
        rule_id: Optional[str] = None,
    ) -> Dict:
        ids = (
            [rule_id]
            if rule_id is not None
            else sorted(
                self.rule_validator.versions
            )
        )

        families = {}
        for rid in ids:
            versions = (
                self.rule_validator.all_versions(
                    rid
                )
            )
            if not versions:
                continue

            families[rid] = [
                {
                    "rule_id": rule.rule_id,
                    "rule_version":
                        rule.rule_version,
                    "rule_instance_id":
                        rule.instance_id,
                    "premises":
                        tuple(rule.premises),
                    "conclusion":
                        rule.conclusion,
                    "context_id":
                        rule.context_id,
                    "valid_from":
                        rule.valid_from,
                    "valid_until":
                        rule.valid_until,
                }
                for rule in versions
            ]

        return {
            "families": families,
            "logical_rule_count":
                len(families),
            "version_count": sum(
                len(items)
                for items in families.values()
            ),
        }

    def supersede_contextual_rule(
        self,
        rule_id: str,
        premises: tuple,
        conclusion: str,
        observed_at: int,
        context_id: Optional[str] = None,
        global_scope: bool = False,
        add_justification: bool = True,
    ) -> Dict:
        """
        Close one active exact-scope version and register its successor.

        This is an explicit lifecycle operation. It never mutates the old
        version's semantics and preserves historical proof before observed_at.
        """
        resolved_context = (
            None
            if global_scope
            else (
                context_id
                if context_id is not None
                else self.belief_contexts.current_id
            )
        )

        active_exact_scope = [
            rule
            for rule
            in self.rule_validator.all_versions(
                rule_id
            )
            if (
                rule.context_id
                == resolved_context
                and rule.applies_to(
                    context_id=resolved_context,
                    as_of=observed_at,
                )[0]
            )
        ]

        if len(active_exact_scope) != 1:
            raise RuleVersionConflict(
                "supersede_contextual_rule memerlukan tepat satu "
                "versi aktif pada exact scope; ditemukan "
                f"{len(active_exact_scope)} untuk {rule_id}"
            )

        previous = active_exact_scope[0]

        closed = (
            self.rule_validator.close_version(
                rule_id,
                previous.rule_version,
                observed_at,
            )
        )

        updated_justifications = []
        for index, justification in enumerate(
            self.justifications
        ):
            if not (
                justification.rule_id
                    == rule_id
                and justification.rule_version
                    == previous.rule_version
                and justification.context_id
                    == resolved_context
            ):
                continue

            if (
                justification.valid_until
                is not None
                and justification.valid_until
                    <= observed_at
            ):
                continue

            closed_justification = replace(
                justification,
                valid_until=observed_at,
            )
            self.justifications[
                index
            ] = closed_justification
            updated_justifications.append(
                closed_justification
            )

        successor = (
            self.add_contextual_rule(
                rule_id=rule_id,
                premises=premises,
                conclusion=conclusion,
                context_id=resolved_context,
                valid_from=observed_at,
                valid_until=None,
                global_scope=global_scope,
                add_justification=add_justification,
            )
        )

        return {
            "operation":
                "supersede_contextual_rule",
            "rule_id": rule_id,
            "context_id":
                resolved_context,
            "observed_at":
                observed_at,
            "previous_rule":
                closed,
            "previous_instance_id":
                closed.instance_id,
            "successor_rule":
                successor,
            "successor_instance_id":
                successor.instance_id,
            "closed_justifications":
                len(updated_justifications),
        }


    def _resolve_belief_scope(
        self,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[str, int]:
        if as_of is None:
            as_of = self.belief_contexts.now

        if context_id is None:
            historical = self.belief_contexts.context_at(as_of)
            context_id = (
                historical.context_id
                if historical is not None
                else self.belief_contexts.current_id
            )

        return context_id, as_of


    def consider_context_shift(
        self,
        claim_id: str,
        incoming_polarity: int,
        observed_at: int,
        reason: str,
        incoming_strength: float = 1.0,
        source: Optional[str] = None,
        origin_id: Optional[str] = None,
        observation_quality: float = 1.0,
    ) -> Dict:
        """
        V2.23 robust context-shift probe.

        It runs BEFORE the caller inserts incoming Evidence. A single
        contradiction normally becomes a pending candidate rather than a new
        context. Persistent contradictions can later confirm a shift.

        Existing callers remain compatible because new arguments are optional.
        """
        current_id = (
            self.belief_contexts.current_id
        )

        if not 0.0 <= observation_quality <= 1.0:
            raise ValueError(
                "observation_quality harus 0..1"
            )

        source_stability = (
            self.source_observation_reliability(
                source
            )
            if source is not None
            else 1.0
        )
        effective_signal_strength = (
            incoming_strength
            * observation_quality
            * source_stability
        )

        # Historical probes are audit-only and must never move current regime.
        historical_signal = (
            observed_at
            < self.interaction_clock
        )

        current_report = self.adjudicate_claim(
            claim_id,
            context_id=current_id,
            as_of=observed_at,
            audit_mode=EvidenceAuditMode.COMPACT,
        )

        if historical_signal:
            assessment = {
                "should_shift": False,
                "pending": False,
                "decision":
                    "historical_signal_ignored",
                "reason":
                    "historical signal cannot revise current belief context",
                "candidate": None,
            }
        else:
            assessment = (
                self.belief_revision_policy
                .assess(
                    context_id=current_id,
                    claim_id=claim_id,
                    current_evidence_status=(
                        current_report[
                            "evidence_status"
                        ]
                    ),
                    incoming_polarity=(
                        incoming_polarity
                    ),
                    observed_at=observed_at,
                    incoming_strength=(
                        effective_signal_strength
                    ),
                    source=source,
                    origin_id=origin_id,
                )
            )

        shifted = bool(
            assessment["should_shift"]
        )

        previous_context = current_id
        if shifted:
            new_context = (
                self.advance_belief_context(
                    observed_at=observed_at,
                    reason=reason,
                )
            )
            current_id = (
                new_context.context_id
            )

        candidate = assessment.get(
            "candidate"
        ) or {}

        self._belief_shift_decision_counter += 1
        audit_record = BeliefShiftDecisionRecord(
            shift_decision_id=(
                self._belief_shift_decision_counter
            ),
            claim_id=claim_id,
            previous_context=previous_context,
            current_context=current_id,
            observed_at=observed_at,
            incoming_polarity=incoming_polarity,
            incoming_strength=effective_signal_strength,
            source=source,
            origin_id=origin_id,
            previous_evidence_status=(
                current_report[
                    "evidence_status"
                ]
            ),
            baseline_evidence_status=(
                candidate.get(
                    "baseline_evidence_status"
                )
            ),
            expected_polarity=(
                candidate.get(
                    "expected_polarity"
                )
            ),
            contradiction_count=int(
                candidate.get(
                    "contradiction_count",
                    0,
                )
            ),
            cumulative_strength=float(
                candidate.get(
                    "cumulative_strength",
                    0.0,
                )
            ),
            independent_signal_count=int(
                candidate.get(
                    "independent_signal_count",
                    0,
                )
            ),
            first_observed_at=(
                candidate.get(
                    "first_observed_at"
                )
            ),
            shifted=shifted,
            pending=bool(
                assessment.get(
                    "pending",
                    False,
                )
            ),
            decision=assessment[
                "decision"
            ],
            reason=assessment[
                "reason"
            ],
        )
        self.belief_shift_memory.append(
            audit_record
        )

        return {
            "shifted": shifted,
            "pending": audit_record.pending,
            "shift_decision_id":
                audit_record.shift_decision_id,
            "claim_id": claim_id,
            "incoming_polarity":
                incoming_polarity,
            "incoming_strength":
                incoming_strength,
            "observation_quality":
                observation_quality,
            "source_observation_reliability":
                source_stability,
            "effective_signal_strength":
                effective_signal_strength,
            "source": source,
            "origin_id": origin_id,
            "previous_context":
                previous_context,
            "current_context":
                current_id,
            "previous_evidence_status":
                current_report[
                    "evidence_status"
                ],
            "observed_at": observed_at,
            "reason": reason,
            "detector_decision":
                audit_record.decision,
            "detector_reason":
                audit_record.reason,
            "candidate": (
                dict(candidate)
                if candidate
                else None
            ),
        }

    def belief_revision_state(
        self,
    ) -> Dict:
        return {
            "policy":
                self.belief_revision_policy.state(),
            "audit_memory":
                self.belief_shift_memory.state(),
            "decision_counter":
                self._belief_shift_decision_counter,
        }


    def touch_interaction_time(
        self,
        observed_at: Optional[int],
    ) -> int:
        """
        Move the actual-interaction clock monotonically.

        Historical timestamps may be supplied and are ignored for clock
        advancement rather than moving time backwards.
        """
        if observed_at is None:
            return self.interaction_clock

        if observed_at >= self.interaction_clock:
            self.interaction_clock = observed_at
            self.belief_contexts.touch(
                observed_at
            )

        return self.interaction_clock

    def advance_interaction_clock(
        self,
        steps: int = 1,
    ) -> int:
        if steps < 1:
            raise ValueError(
                "steps harus >= 1"
            )

        self.interaction_clock += steps
        self.belief_contexts.touch(
            self.interaction_clock
        )
        return self.interaction_clock

    def advance_belief_context(
        self,
        observed_at: int,
        reason: str,
    ) -> BeliefContext:
        """
        Start a new belief epoch without deleting previous evidence.
        """
        self.touch_interaction_time(
            observed_at
        )
        previous_context = (
            self.belief_contexts.current_id
        )
        new_context = (
            self.belief_contexts.advance(
                observed_at,
                reason,
            )
        )
        if hasattr(
            self.belief_revision_policy,
            "clear_context",
        ):
            self.belief_revision_policy.clear_context(
                previous_context
            )
        return new_context

    def add_contextual_evidence(
        self,
        evidence_id: str,
        source: str,
        origin_id: str,
        claim_id: str,
        polarity: int,
        strength: float,
        observed_at: Optional[int] = None,
        context_id: Optional[str] = None,
        valid_from: Optional[int] = None,
        valid_until: Optional[int] = None,
        global_scope: bool = False,
        observation_quality: float = 1.0,
        retry_group_id: Optional[str] = None,
    ) -> Evidence:
        """
        Convenience API for scoped evidence.

        global_scope=True makes evidence cross-context.
        Otherwise missing context_id resolves to the current belief context.
        """
        self.touch_interaction_time(observed_at)

        resolved_context = (
            None
            if global_scope
            else (
                context_id
                if context_id is not None
                else self.belief_contexts.current_id
            )
        )

        evidence = Evidence(
            evidence_id=evidence_id,
            source=source,
            origin_id=origin_id,
            claim_id=claim_id,
            polarity=polarity,
            strength=strength,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_until=valid_until,
            context_id=resolved_context,
            observation_quality=observation_quality,
            retry_group_id=retry_group_id,
        )
        self.evidence_pool.append(evidence)
        self._evidence_revision += 1
        self._update_observation_reliability(
            evidence
        )
        self.maintain_epistemic_archive()
        return evidence

    def belief_context_state(self) -> Dict:
        return {
            "now": self.belief_contexts.now,
            "current_context": self.belief_contexts.current_id,
            "contexts": self.belief_contexts.state(),
            "contextual_admission": {
                f"{context}|{claim}": status.value
                for (context, claim), status
                in self.contextual_admission.items()
            },
        }

    def adjudicate_claim(
        self,
        claim_id: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        audit_mode: EvidenceAuditMode = EvidenceAuditMode.FULL,
    ) -> Dict:
        context_id, as_of = self._resolve_belief_scope(
            context_id=context_id,
            as_of=as_of,
        )

        active_grounded = self.active_grounded(
            context_id=context_id,
            as_of=as_of,
        )

        truth_status, used_axioms = self.truth_evaluator.evaluate(
            claim_id,
            self.justifications,
            active_grounded,
            context_id=context_id,
            as_of=as_of,
        )
        if isinstance(
            audit_mode,
            str,
        ):
            audit_mode = (
                EvidenceAuditMode(
                    audit_mode
                )
            )

        indexed_audit = None
        indexed_cache_hit = False

        if (
            audit_mode
            == EvidenceAuditMode.FULL
        ):
            ev_status, s_score, o_score = (
                self.evidence_aggregator.aggregate(
                    claim_id,
                    self._evidence_for_claim_exact(
                        claim_id
                    ),
                    context_id=context_id,
                    as_of=as_of,
                )
            )
        else:
            (
                indexed_audit,
                indexed_cache_hit,
            ) = (
                self.evidence_query_engine
                .aggregate(
                    claim_id,
                    context_id,
                    as_of,
                )
            )
            ev_status = (
                indexed_audit.status
            )
            s_score = (
                indexed_audit.support_score
            )
            o_score = (
                indexed_audit.oppose_score
            )

            # Compact mode intentionally does not materialize forensic
            # evidence-detail lists. The score/verdict remain exact.
            self.evidence_aggregator.last_dropped = []
            self.evidence_aggregator.last_out_of_scope = []

        # V2.1: ontology adalah PRIOR terpisah, bukan empirical evidence.
        ontology_strength = 0.0
        if claim_id in self.kb.concepts:
            for g in active_grounded:
                if g in self.kb.concepts:
                    ontology_strength = max(
                        ontology_strength,
                        self.kb.relation_strength(g, claim_id),
                    )
        ontology_prior = ontology_strength * ONTOLOGY_WEIGHT

        if truth_status == "invalid":
            verdict = EpistemicVerdict.LOGICAL_FALLACY
        elif ev_status == "conflicted":
            verdict = EpistemicVerdict.EPISTEMIC_CRISIS
        elif truth_status == "supported" and ev_status == "accepted":
            verdict = EpistemicVerdict.VERIFIED_FACT
        elif truth_status == "supported" and ev_status == "rejected":
            verdict = EpistemicVerdict.EMPIRICAL_ANOMALY
        elif truth_status == "supported" and ev_status == "unresolved":
            verdict = EpistemicVerdict.THEORETICAL_CONSTRUCT
        elif truth_status == "unknown" and ev_status == "accepted":
            verdict = EpistemicVerdict.EMPIRICAL_DISCOVERY
        elif truth_status == "unknown" and ev_status == "rejected":
            verdict = EpistemicVerdict.EMPIRICAL_REFUTATION
        else:
            verdict = EpistemicVerdict.UNRESOLVED

        # PERBAIKAN #2: aksioma tanpa provenance ditandai secara eksplisit,
        # termasuk klaim itu sendiri bila ia grounded tanpa asal-usul tercatat
        axiom_provenance = {}
        unaudited_axioms = set()

        for axiom in used_axioms:
            scoped = self.grounding_store.active_provenance(
                axiom,
                context_id=context_id,
                as_of=as_of,
            )

            if scoped:
                axiom_provenance[axiom] = scoped
            elif axiom in self.grounded_provenance:
                axiom_provenance[axiom] = [
                    self.grounded_provenance[axiom]
                ]
            else:
                axiom_provenance[axiom] = []
                unaudited_axioms.add(axiom)

        if claim_id in active_grounded and claim_id not in axiom_provenance:
            scoped = self.grounding_store.active_provenance(
                claim_id,
                context_id=context_id,
                as_of=as_of,
            )
            if not scoped and claim_id not in self.grounded_provenance:
                unaudited_axioms.add(claim_id)

        return {
            "claim_id": claim_id,
            "belief_context_id": context_id,
            "as_of": as_of,
            "verdict": verdict,
            "truth_status": truth_status,
            "evidence_status": ev_status,
            "support_score": s_score,
            "oppose_score": o_score,
            "used_axioms": used_axioms,
            "ontology_strength": ontology_strength,
            "ontology_prior": ontology_prior,
            "empirical_support_score": s_score,
            "axiom_provenance": axiom_provenance,
            "unaudited_axioms": unaudited_axioms,
            # PERBAIKAN #5: bukti yang dibuang kini ikut dilaporkan
            "dropped_evidence": list(self.evidence_aggregator.last_dropped),
            "retry_quarantined_groups": list(
                self.evidence_aggregator.last_retry_quarantined_groups
            ),
            "out_of_scope_evidence": list(
                self.evidence_aggregator.last_out_of_scope
            ),
            "active_grounded": set(active_grounded),
            "out_of_scope_groundings": list(
                self.grounding_store.last_out_of_scope
            ),
            "out_of_scope_proof": list(
                self.truth_evaluator.last_out_of_scope_proof
            ),
            "evidence_audit_mode":
                audit_mode.value,
            "evidence_query": (
                None
                if indexed_audit is None
                else {
                    "cache_hit":
                        indexed_cache_hit,
                    "total_records":
                        indexed_audit.total_records,
                    "in_scope_records":
                        indexed_audit.in_scope_records,
                    "out_of_scope_count":
                        indexed_audit.out_of_scope_count,
                    "unknown_source_count":
                        indexed_audit.unknown_source_count,
                    "duplicate_drop_count":
                        indexed_audit.duplicate_drop_count,
                    "source_candidate_count":
                        indexed_audit.source_candidate_count,
                    "origin_winner_count":
                        indexed_audit.origin_winner_count,
                    "cold_candidate_rows":
                        indexed_audit.cold_candidate_rows,
                    "hot_records_scanned":
                        indexed_audit.hot_records_scanned,
                    "aggregate_below_min_count":
                        indexed_audit.aggregate_below_min_count,
                    "retry_quarantined_group_count":
                        indexed_audit.retry_quarantined_group_count,
                }
            ),
        }


    def _apply_admission(
        self,
        claim_id: str,
        status: AdmissionStatus,
        context_id: Optional[str] = None,
    ):
        """
        Store admission by (belief context, claim).

        Legacy flat sets are kept as the CURRENT-context projection.
        """
        resolved_context = (
            context_id
            if context_id is not None
            else self.belief_contexts.current_id
        )
        self.contextual_admission[
            (resolved_context, claim_id)
        ] = status

        if resolved_context != self.belief_contexts.current_id:
            return

        self.accepted_claims.discard(claim_id)
        self.pending_claims.discard(claim_id)
        self.quarantined_claims.discard(claim_id)
        self.rejected_claims.discard(claim_id)

        if status == AdmissionStatus.ACCEPTED:
            self.accepted_claims.add(claim_id)
        elif status == AdmissionStatus.PENDING:
            self.pending_claims.add(claim_id)
        elif status == AdmissionStatus.QUARANTINED:
            self.quarantined_claims.add(claim_id)
        elif status == AdmissionStatus.REJECTED:
            self.rejected_claims.add(claim_id)

    def observe_claim(
        self,
        claim_id: str,
        notes: str = "",
        context_id: Optional[str] = None,
        observed_at: Optional[int] = None,
    ) -> Episode:
        """
        Scoped cognitive episode:
        adjudicate -> contextual admission -> memory.

        Tidak mengubah grounded facts.
        """
        self.touch_interaction_time(observed_at)
        report = self.adjudicate_claim(
            claim_id,
            context_id=context_id,
            as_of=observed_at,
            audit_mode=EvidenceAuditMode.COMPACT,
        )
        admission = self.admission_policy.decide(report["verdict"])
        self._apply_admission(
            claim_id,
            admission,
            context_id=report["belief_context_id"],
        )

        self._episode_counter += 1
        episode = Episode(
            episode_id=self._episode_counter,
            claim_id=claim_id,
            verdict=report["verdict"],
            truth_status=report["truth_status"],
            evidence_status=report["evidence_status"],
            support_score=report["support_score"],
            oppose_score=report["oppose_score"],
            selected_proof=list(self.truth_evaluator.last_selected_proof),
            used_axioms=set(report["used_axioms"]),
            admission_status=admission,
            notes=notes,
            belief_context_id=report["belief_context_id"],
            observed_at=report["as_of"],
        )
        self.memory.append(episode)
        self.maintain_epistemic_archive()
        return episode

    def record_episode_outcome(
        self,
        episode_id: int,
        accurate: bool,
        feedback_weight: float = 1.0,
    ):
        """
        Hubungkan outcome dunia ke episode dan update reliabilitas source.

        Evidence dengan claim_id yang sama diberi feedback.
        origin_id yang sama hanya memberi feedback satu kali agar copy tidak
        menggandakan pembelajaran source.
        """
        target = self.memory.get(
            episode_id
        )
        if target is None:
            raise KeyError(
                f"Episode {episode_id} tidak ditemukan"
            )

        self.memory.set_outcome(
            episode_id,
            accurate,
        )
        target.outcome = accurate

        scoped: List[Evidence] = []
        for e in self._evidence_for_claim_exact(
            target.claim_id
        ):
            in_scope, _ = e.applies_to(
                context_id=target.belief_context_id,
                as_of=target.observed_at,
            )
            if in_scope:
                scoped.append(e)

        retry_groups: Dict[str, List[Evidence]] = {}
        legacy: List[Evidence] = []
        for e in scoped:
            if e.retry_group_id is None:
                legacy.append(e)
            else:
                retry_groups.setdefault(
                    e.retry_group_id,
                    [],
                ).append(e)

        # Factual source feedback must not learn from a retry group that the
        # observation layer itself classified as operationally inconsistent.
        eligible: List[Evidence] = list(legacy)
        for _group_id, items in retry_groups.items():
            if len({e.polarity for e in items}) > 1:
                continue
            eligible.append(
                max(
                    items,
                    key=lambda e: (
                        e.strength
                        * e.observation_quality
                    ),
                )
            )

        best_by_origin: Dict[str, Evidence] = {}
        for e in eligible:
            current = best_by_origin.get(e.origin_id)
            current_value = (
                -1.0
                if current is None
                else (
                    current.strength
                    * current.observation_quality
                )
            )
            value = (
                e.strength
                * e.observation_quality
            )
            if current is None or value > current_value:
                best_by_origin[e.origin_id] = e

        updated_sources = set()
        for e in best_by_origin.values():
            if e.source in updated_sources:
                continue
            profile = self.sources.get(e.source)
            if profile is None:
                continue

            # Jika evidence menolak claim, outcome claim yang benar berarti
            # evidence tersebut tidak akurat, dan sebaliknya.
            evidence_accurate = accurate if e.polarity == 1 else not accurate
            profile.record_feedback(
                evidence_accurate,
                weight=(
                    feedback_weight
                    * e.strength
                    * e.observation_quality
                ),
            )
            updated_sources.add(e.source)

        return {
            "episode_id": episode_id,
            "claim_id": target.claim_id,
            "outcome": accurate,
            "updated_sources": updated_sources,
        }

    def retract_grounded(self, claim: str, reason: str = "") -> Dict:
        """
        Legacy GLOBAL retract. Use retract_contextual_grounded() for scoped
        temporal facts.
        """
        return self.tms.retract(
            claim,
            reason=reason,
        )

    def retract_contextual_grounded(
        self,
        claim: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        reason: str = "",
    ) -> Dict:
        return self.tms.retract_contextual(
            claim=claim,
            context_id=context_id,
            as_of=as_of,
            reason=reason,
        )

    def restore_contextual_grounded(
        self,
        claim: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        origin: str = "contextual_restore",
        note: str = "",
    ) -> Dict:
        return self.tms.restore_contextual(
            claim=claim,
            context_id=context_id,
            as_of=as_of,
            origin=origin,
            note=note,
        )

    def restore_grounded(
        self,
        claim: str,
        origin: str = "manual_restore",
        note: str = "",
    ) -> Dict:
        """Pulihkan grounded fact dan reevaluate descendant secara transitif."""
        return self.tms.restore_grounded(
            claim,
            origin=origin,
            note=note,
        )

    def learning_state(self) -> Dict:
        return {
            "episodes": len(self.memory.all()),
            "accepted": set(self.accepted_claims),
            "pending": set(self.pending_claims),
            "quarantined": set(self.quarantined_claims),
            "rejected": set(self.rejected_claims),
        }


    def choose_action_preference_aware(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[
            Dict[str, float]
        ] = None,
        belief_context_id: Optional[str] = None,
    ) -> DecisionRecord:
        """
        Preference-shift-safe cold-start decision.

        Priority order for the utility channel:
        1. Q learned under the CURRENT objective-profile version, if sampled.
        2. Read-only reweighted ACTUAL objective-vector history.
        3. Initial neutral Q prior.

        Reweighted vector history is never written into Q. It is only a
        decision-time interpretation of measurements already observed.
        """
        self.maintain_memory(
            (
                "decision",
                "transition",
            )
        )

        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )
        profile = self.objective_profile
        identities = (
            self._resolve_action_candidates(
                candidates
            )
        )
        references = list(
            identities
        )

        epistemic_scores = (
            epistemic_scores or {}
        )
        unknown = (
            set(epistemic_scores)
            - set(references)
        )
        if unknown:
            raise ValueError(
                "epistemic_scores memiliki action "
                f"yang bukan kandidat: {sorted(unknown)}"
            )

        utilities = {}
        utility_sources = {}
        q_counts = {}
        reweighted_support = {}
        epistemics = {}
        explorations = {}
        policy_scores = {}
        audits = {}

        for reference in references:
            identity = identities[
                reference
            ]
            instance = (
                identity.instance_id
            )
            q_count = (
                self.decision_policy.count(
                    scalar_state_key,
                    instance,
                    belief_context_id=scope,
                )
            )
            q_value = (
                self.decision_policy.utility(
                    scalar_state_key,
                    instance,
                    belief_context_id=scope,
                )
            )
            reweighted = (
                self.reweighted_objective_estimate(
                    context,
                    reference,
                    objective_profile_reference=(
                        profile.instance_id
                    ),
                    belief_context_id=scope,
                )
            )

            if q_count > 0:
                effective_utility = q_value
                source = "profile_q"
            elif (
                reweighted["utility"]
                is not None
            ):
                effective_utility = (
                    reweighted[
                        "utility"
                    ]
                )
                source = (
                    "reweighted_actual_vector_history"
                )
            else:
                effective_utility = q_value
                source = "neutral_prior"

            epistemic = (
                epistemic_scores.get(
                    reference,
                    0.5,
                )
            )
            if not 0.0 <= epistemic <= 1.0:
                raise ValueError(
                    "epistemic score harus 0..1"
                )

            exploration = (
                self.decision_policy
                .exploration_bonus(
                    scalar_state_key,
                    instance,
                    len(references),
                    belief_context_id=scope,
                )
            )

            score = (
                self.decision_policy.utility_weight
                * effective_utility
                + self.decision_policy.epistemic_weight
                * epistemic
                + self.decision_policy.exploration_weight
                * exploration
            )

            utilities[reference] = (
                effective_utility
            )
            utility_sources[reference] = (
                source
            )
            q_counts[reference] = (
                q_count
            )
            reweighted_support[
                reference
            ] = (
                reweighted["support"]
            )
            epistemics[reference] = (
                epistemic
            )
            explorations[reference] = (
                exploration
            )
            policy_scores[reference] = (
                score
            )
            audits[reference] = {
                "utility_source":
                    source,
                "profile_q_count":
                    q_count,
                "profile_q_value":
                    q_value,
                "reweighted_utility":
                    reweighted[
                        "utility"
                    ],
                "reweighted_support":
                    reweighted[
                        "support"
                    ],
                "reweighted_std":
                    reweighted[
                        "std"
                    ],
                "reweighted_variance":
                    reweighted[
                        "variance"
                    ],
                "reweighted_coverage":
                    reweighted[
                        "coverage"
                    ],
                "reweighted_unscorable_count":
                    reweighted[
                        "unscorable_count"
                    ],
                "reweighted_mask_count":
                    reweighted[
                        "mask_count"
                    ],
                "objective_profile_instance_id":
                    profile.instance_id,
            }

        selected = max(
            references,
            key=lambda action: (
                policy_scores[action],
                action,
            ),
        )
        selected_identity = (
            identities[
                selected
            ]
        )

        self._decision_counter += 1
        record = DecisionRecord(
            decision_id=(
                self._decision_counter
            ),
            context=context,
            candidates=tuple(
                references
            ),
            selected_action=selected,
            policy_scores=dict(
                policy_scores
            ),
            utility_estimates=dict(
                utilities
            ),
            epistemic_scores=dict(
                epistemics
            ),
            exploration_scores=dict(
                explorations
            ),
            belief_context_id=scope,
            selection_mode=(
                "preference_aware"
            ),
            strategy_scores=dict(
                policy_scores
            ),
            counterfactual_rewards={
                action:
                    (
                        audits[action][
                            "reweighted_utility"
                        ]
                    )
                for action in references
                if audits[action][
                    "reweighted_utility"
                ] is not None
            },
            uncertainty_audit={
                action:
                    dict(values)
                for action, values
                in audits.items()
            },
            candidate_action_instances={
                reference:
                    identity.instance_id
                for reference, identity
                in identities.items()
            },
            selected_action_family=(
                selected_identity.family
            ),
            selected_action_instance_id=(
                selected_identity.instance_id
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                profile.instance_id
            ),
            objective_profile_signature=(
                profile.signature
            ),
        )
        self.decision_memory.append(
            record
        )
        self.maintain_memory(
            ("decision",)
        )
        return record



    def choose_action(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[Dict[str, float]] = None,
        belief_context_id: Optional[str] = None,
    ) -> DecisionRecord:
        """
        V2.22 compatibility:
        public candidate names stay unchanged, while registered actions use
        immutable instance IDs as Q-learning keys.
        """
        self.maintain_memory(
            ("decision", "transition", "meta_risk")
        )

        resolved_belief_context = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )
        identities = self._resolve_action_candidates(
            candidates
        )
        references = list(identities)
        instance_candidates = [
            identities[reference].instance_id
            for reference in references
        ]

        instance_epistemic = (
            self._to_instance_mapping(
                epistemic_scores,
                identities,
            )
        )

        (
            selected_instance,
            policy_scores_i,
            utility_estimates_i,
            epistemic_i,
            exploration_i,
        ) = self.decision_policy.select(
            scalar_state_key,
            instance_candidates,
            instance_epistemic,
            belief_context_id=(
                resolved_belief_context
            ),
        )

        reverse = {
            identity.instance_id: reference
            for reference, identity
            in identities.items()
        }
        selected_reference = reverse[
            selected_instance
        ]
        selected_identity = identities[
            selected_reference
        ]

        self._decision_counter += 1
        record = DecisionRecord(
            decision_id=self._decision_counter,
            context=context,
            candidates=tuple(references),
            selected_action=selected_reference,
            policy_scores=self._translate_instance_dict(
                policy_scores_i,
                identities,
            ),
            utility_estimates=self._translate_instance_dict(
                utility_estimates_i,
                identities,
            ),
            epistemic_scores=self._translate_instance_dict(
                epistemic_i,
                identities,
            ),
            exploration_scores=self._translate_instance_dict(
                exploration_i,
                identities,
            ),
            belief_context_id=resolved_belief_context,
            candidate_action_instances={
                reference: identity.instance_id
                for reference, identity
                in identities.items()
            },
            selected_action_family=(
                selected_identity.family
            ),
            selected_action_instance_id=(
                selected_identity.instance_id
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
            objective_profile_signature=(
                self.objective_profile.signature
            ),
        )

        self.decision_memory.append(record)
        self.maintain_memory(
            ("decision",)
        )
        return record

    def choose_action_adaptive_risk(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[
            Dict[str, float]
        ] = None,
        failure_consequences: Optional[
            Dict[str, float]
        ] = None,
        belief_context_id: Optional[str] = None,
    ) -> DecisionRecord:
        """
        Adaptive risk decision with action-version-isolated Q/world-model
        learning. Public action references remain stable for adapters.
        """
        if not candidates:
            raise ValueError(
                "candidates tidak boleh kosong"
            )

        self.maintain_memory(
            (
                "decision",
                "transition",
                "prediction",
                "prediction_error",
                "meta_risk",
            )
        )

        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )
        identities = self._resolve_action_candidates(
            candidates
        )
        references = list(identities)
        instance_candidates = [
            identities[reference].instance_id
            for reference in references
        ]

        consequences = {
            action: 0.50
            for action in references
        }
        if failure_consequences is not None:
            unknown = (
                set(failure_consequences)
                - set(references)
            )
            if unknown:
                raise ValueError(
                    "failure_consequences memiliki action "
                    f"yang bukan kandidat: {sorted(unknown)}"
                )
            for action, value in (
                failure_consequences.items()
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        "failure consequence harus 0..1"
                    )
                consequences[action] = value

        (
            base_i,
            utilities_i,
            epistemic_i,
            exploration_i,
        ) = self.decision_policy.score_actions(
            scalar_state_key,
            instance_candidates,
            self._to_instance_mapping(
                epistemic_scores,
                identities,
            ),
            belief_context_id=resolved_scope,
        )

        base_scores = self._translate_instance_dict(
            base_i,
            identities,
        )
        utilities = self._translate_instance_dict(
            utilities_i,
            identities,
        )
        normalized_epistemic = (
            self._translate_instance_dict(
                epistemic_i,
                identities,
            )
        )
        exploration_scores = (
            self._translate_instance_dict(
                exploration_i,
                identities,
            )
        )

        predictions = {
            action: self.predict_outcome(
                context,
                action,
                belief_context_id=resolved_scope,
            )
            for action in references
        }

        signals = (
            self.adaptive_risk_mode_policy.signals(
                context=context,
                predictions=predictions,
                failure_consequences=consequences,
                belief_context_id=resolved_scope,
                base_scores=base_scores,
            )
        )
        mode, reason = (
            self.adaptive_risk_mode_policy.select_mode(
                signals
            )
        )
        profile = (
            self.adaptive_risk_mode_policy.profile_for_mode(
                mode
            )
        )
        result = self.uncertainty_decision_policy.select(
            base_scores=base_scores,
            predictions=predictions,
            profile=profile,
        )

        self._meta_decision_counter += 1
        meta_record = MetaRiskDecision(
            meta_decision_id=self._meta_decision_counter,
            belief_context_id=resolved_scope,
            context=context,
            selected_mode=mode,
            signals=signals,
            reason=reason,
            candidate_actions=tuple(references),
            candidate_action_instances={
                reference: identity.instance_id
                for reference, identity
                in identities.items()
            },
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
        )
        self.meta_risk_memory.append(
            meta_record
        )
        self.maintain_memory(
            ("meta_risk",)
        )

        selected_identity = identities[
            result.selected_action
        ]

        self._decision_counter += 1
        record = DecisionRecord(
            decision_id=self._decision_counter,
            context=context,
            candidates=tuple(references),
            selected_action=result.selected_action,
            policy_scores=dict(base_scores),
            utility_estimates=dict(utilities),
            epistemic_scores=dict(
                normalized_epistemic
            ),
            exploration_scores=dict(
                exploration_scores
            ),
            belief_context_id=resolved_scope,
            selection_mode="adaptive_risk",
            strategy_scores=dict(
                result.decision_scores
            ),
            counterfactual_rewards={
                action:
                    prediction.predicted_reward
                for action, prediction
                in predictions.items()
            },
            risk_mode=mode.value,
            uncertainty_scores=dict(
                result.uncertainty_scores
            ),
            blocked_actions=result.blocked_actions,
            prediction_ids={
                action:
                    prediction.prediction_id
                for action, prediction
                in predictions.items()
            },
            meta_decision_id=(
                meta_record.meta_decision_id
            ),
            candidate_action_instances={
                reference: identity.instance_id
                for reference, identity
                in identities.items()
            },
            selected_action_family=(
                selected_identity.family
            ),
            selected_action_instance_id=(
                selected_identity.instance_id
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
            objective_profile_signature=(
                self.objective_profile.signature
            ),
        )
        self.decision_memory.append(record)

        selected_prediction_id = (
            record.prediction_ids[
                record.selected_action
            ]
        )
        self._active_prediction_pins.add(
            selected_prediction_id
        )

        self.maintain_memory(
            ("decision", "prediction")
        )
        return record

    def adaptive_risk_state(
        self,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        records = (
            self.meta_risk_memory.for_belief_context(
                resolved_scope
            )
        )

        counts = {
            mode.value: 0
            for mode in UncertaintyDecisionMode
        }
        for record in records:
            counts[record.selected_mode.value] += 1

        return {
            "belief_context_id": resolved_scope,
            "meta_decisions": len(records),
            "mode_counts": counts,
            "last": (
                records[-1]
                if records
                else None
            ),
        }

    def _choose_action_uncertainty_aware_v221(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[
            Dict[str, float]
        ] = None,
        risk_profile: Optional[
            UncertaintyRiskProfile
        ] = None,
        belief_context_id: Optional[str] = None,
    ) -> DecisionRecord:
        """
        V2.21 uncertainty policy with V2.22 action-instance isolation.
        """
        if not candidates:
            raise ValueError(
                "candidates tidak boleh kosong"
            )

        self.maintain_memory(
            (
                "decision",
                "transition",
                "prediction",
                "prediction_error",
                "meta_risk",
            )
        )

        profile = (
            risk_profile
            if risk_profile is not None
            else UncertaintyRiskProfile()
        )
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )

        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )
        identities = self._resolve_action_candidates(
            candidates
        )
        references = list(identities)
        instance_candidates = [
            identities[reference].instance_id
            for reference in references
        ]

        (
            base_i,
            utilities_i,
            epistemic_i,
            exploration_i,
        ) = self.decision_policy.score_actions(
            scalar_state_key,
            instance_candidates,
            self._to_instance_mapping(
                epistemic_scores,
                identities,
            ),
            belief_context_id=resolved_scope,
        )
        base_scores = self._translate_instance_dict(
            base_i,
            identities,
        )
        utilities = self._translate_instance_dict(
            utilities_i,
            identities,
        )
        normalized_epistemic = (
            self._translate_instance_dict(
                epistemic_i,
                identities,
            )
        )
        exploration_scores = (
            self._translate_instance_dict(
                exploration_i,
                identities,
            )
        )

        predictions = {
            action: self.predict_outcome(
                context,
                action,
                belief_context_id=resolved_scope,
            )
            for action in references
        }

        result = self.uncertainty_decision_policy.select(
            base_scores=base_scores,
            predictions=predictions,
            profile=profile,
        )
        selected_identity = identities[
            result.selected_action
        ]

        self._decision_counter += 1
        record = DecisionRecord(
            decision_id=self._decision_counter,
            context=context,
            candidates=tuple(references),
            selected_action=result.selected_action,
            policy_scores=dict(base_scores),
            utility_estimates=dict(utilities),
            epistemic_scores=dict(
                normalized_epistemic
            ),
            exploration_scores=dict(
                exploration_scores
            ),
            belief_context_id=resolved_scope,
            selection_mode=(
                f"uncertainty_{profile.mode.value}"
            ),
            strategy_scores=dict(
                result.decision_scores
            ),
            counterfactual_rewards={
                action:
                    prediction.predicted_reward
                for action, prediction
                in predictions.items()
            },
            risk_mode=profile.mode.value,
            uncertainty_scores=dict(
                result.uncertainty_scores
            ),
            blocked_actions=result.blocked_actions,
            prediction_ids={
                action:
                    prediction.prediction_id
                for action, prediction
                in predictions.items()
            },
            candidate_action_instances={
                reference: identity.instance_id
                for reference, identity
                in identities.items()
            },
            selected_action_family=(
                selected_identity.family
            ),
            selected_action_instance_id=(
                selected_identity.instance_id
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
            objective_profile_signature=(
                self.objective_profile.signature
            ),
        )
        self.decision_memory.append(record)

        selected_prediction_id = (
            record.prediction_ids[
                record.selected_action
            ]
        )
        self._active_prediction_pins.add(
            selected_prediction_id
        )

        self.maintain_memory(
            ("decision", "prediction")
        )
        return record

    def choose_trajectory_uncertainty_aware(
        self,
        context: str,
        candidate_trajectories: Dict[
            str,
            Tuple[str, ...],
        ],
        trajectory_estimates: Dict[
            str,
            TrajectoryRiskEstimate,
        ],
        base_scores: Dict[str, float],
        belief_context_id: Optional[str] = None,
        model_reliability: float = 1.0,
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        """
        V3.2 trajectory ranking with exact V2.22 action-version provenance.

        Registered actions require each step ActionRiskEstimate to carry the
        matching action_instance_id. This prevents a trajectory estimate built
        for v1 from being reused after the family has moved to v2.
        """
        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )

        if not candidate_trajectories:
            raise ValueError(
                "candidate_trajectories tidak boleh kosong"
            )
        ids = set(candidate_trajectories)
        if (
            ids != set(trajectory_estimates)
            or ids != set(base_scores)
        ):
            raise ValueError(
                "candidate, estimate, dan base score trajectory harus sama"
            )

        normalized_candidates = {}
        instance_candidates = {}

        for (
            trajectory_id,
            actions,
        ) in candidate_trajectories.items():
            normalized = tuple(actions)
            if not normalized:
                raise ValueError(
                    "Candidate trajectory tidak boleh kosong"
                )

            estimate = trajectory_estimates[
                trajectory_id
            ]
            if (
                estimate.trajectory_id
                != trajectory_id
            ):
                raise ValueError(
                    "trajectory_id estimate tidak konsisten"
                )
            if estimate.actions != normalized:
                raise ValueError(
                    "Urutan action candidate dan estimate tidak konsisten"
                )

            action_instances = []
            for reference, step in zip(
                normalized,
                estimate.steps,
            ):
                identity = (
                    self._validate_action_estimate_identity(
                        reference,
                        step,
                        context=context,
                    )
                )
                action_instances.append(
                    identity.instance_id
                )

            normalized_candidates[
                trajectory_id
            ] = normalized
            instance_candidates[
                trajectory_id
            ] = tuple(
                action_instances
            )

        ranking = self.trajectory_risk_policy.rank(
            base_scores,
            trajectory_estimates,
            model_reliability=model_reliability,
            requested_mode=requested_mode,
        )
        selected_id = ranking[
            "selected_trajectory_id"
        ]

        if selected_id is None:
            return {
                "decision": None,
                "ranking": ranking,
            }

        selected_actions = (
            normalized_candidates[
                selected_id
            ]
        )
        selected_instances = (
            instance_candidates[
                selected_id
            ]
        )

        self._trajectory_decision_counter += 1
        record = TrajectoryDecisionRecord(
            trajectory_decision_id=(
                self._trajectory_decision_counter
            ),
            context=context,
            belief_context_id=(
                belief_context_id
            ),
            candidate_trajectories={
                trajectory_id:
                    tuple(actions)
                for trajectory_id, actions
                in normalized_candidates.items()
            },
            selected_trajectory_id=(
                selected_id
            ),
            selected_actions=(
                selected_actions
            ),
            selected_first_action=(
                selected_actions[0]
            ),
            risk_mode=ranking[
                "risk_mode"
            ].value,
            base_scores=dict(
                base_scores
            ),
            risk_adjusted_scores=dict(
                ranking[
                    "risk_adjusted_scores"
                ]
            ),
            trajectory_audit={
                trajectory_id:
                    dict(values)
                for trajectory_id, values
                in ranking[
                    "trajectory_audit"
                ].items()
            },
            candidate_trajectory_action_instances={
                trajectory_id:
                    tuple(instance_ids)
                for trajectory_id,
                    instance_ids
                in instance_candidates.items()
            },
            selected_action_instance_ids=(
                selected_instances
            ),
            selected_first_action_instance_id=(
                selected_instances[0]
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
        )
        self.trajectory_decision_memory.append(
            record
        )

        return {
            "decision": record,
            "ranking": ranking,
        }

    def _choose_action_uncertainty_aware_v32(
        self,
        context: str,
        candidates: List[str],
        action_estimates: Dict[str, ActionRiskEstimate],
        epistemic_scores: Optional[Dict[str, float]] = None,
        belief_context_id: Optional[str] = None,
        model_reliability: float = 1.0,
        requested_mode: Optional[RiskMode] = None,
        trajectory_decision_id: Optional[int] = None,
        selected_trajectory_id: Optional[str] = None,
        planned_actions: Tuple[str, ...] = (),
        trajectory_failure_bounds: Optional[
            Tuple[float, float]
        ] = None,
    ) -> Dict:
        """
        V3.2 risk ranking with V2.22 version-safe utility provenance.

        Registered ActionRiskEstimate records MUST name the action instance
        they were computed from; stale estimates are rejected.
        """
        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )
        identities = self._resolve_action_candidates(
            candidates
        )
        unique_candidates = list(
            identities
        )
        if set(unique_candidates) != set(
            action_estimates
        ):
            raise ValueError(
                "action_estimates harus tepat mencakup semua candidates"
            )

        for reference in unique_candidates:
            self._validate_action_estimate_identity(
                reference,
                action_estimates[reference],
                context=context,
            )

        if planned_actions:
            if planned_actions[0] not in unique_candidates:
                raise ValueError(
                    "Action pertama trajectory harus menjadi candidate"
                )
            if trajectory_decision_id is None:
                raise ValueError(
                    "planned_actions membutuhkan trajectory_decision_id"
                )
        if trajectory_failure_bounds is not None:
            lower, upper = trajectory_failure_bounds
            if not (0.0 <= lower <= upper <= 1.0):
                raise ValueError(
                    "trajectory_failure_bounds tidak valid"
                )

        instance_candidates = [
            identities[reference].instance_id
            for reference in unique_candidates
        ]

        (
            base_i,
            utility_i,
            epistemic_i,
            exploration_i,
        ) = self.decision_policy.score_actions(
            scalar_state_key,
            instance_candidates,
            self._to_instance_mapping(
                epistemic_scores,
                identities,
            ),
            belief_context_id=belief_context_id,
        )
        base_scores = self._translate_instance_dict(
            base_i,
            identities,
        )
        utilities = self._translate_instance_dict(
            utility_i,
            identities,
        )
        normalized_epistemic = (
            self._translate_instance_dict(
                epistemic_i,
                identities,
            )
        )
        explorations = self._translate_instance_dict(
            exploration_i,
            identities,
        )

        ranking = self.uncertainty_decision_policy.rank(
            base_scores,
            action_estimates,
            model_reliability=model_reliability,
            requested_mode=requested_mode,
        )
        selected = ranking["selected_action"]

        if selected is None:
            return {
                "decision": None,
                "ranking": ranking,
                "base_policy_scores":
                    base_scores,
                "utility_estimates":
                    utilities,
                "epistemic_scores":
                    normalized_epistemic,
                "exploration_scores":
                    explorations,
            }

        selected_identity = identities[
            selected
        ]
        planned_instance_ids = tuple(
            self.resolve_action_identity(
                action,
                require_active=True,
            ).instance_id
            for action in planned_actions
        )

        self._decision_counter += 1
        record = DecisionRecord(
            decision_id=self._decision_counter,
            context=context,
            candidates=tuple(unique_candidates),
            selected_action=selected,
            policy_scores=dict(base_scores),
            utility_estimates=dict(utilities),
            epistemic_scores=dict(
                normalized_epistemic
            ),
            exploration_scores=dict(
                explorations
            ),
            selection_mode="uncertainty_aware",
            strategy_scores=dict(
                ranking["risk_adjusted_scores"]
            ),
            counterfactual_rewards={
                action:
                    action_estimates[
                        action
                    ].predicted_reward
                for action
                in unique_candidates
            },
            belief_context_id=belief_context_id,
            risk_mode=ranking[
                "risk_mode"
            ].value,
            risk_adjusted_scores=dict(
                ranking["risk_adjusted_scores"]
            ),
            uncertainty_audit={
                action: dict(values)
                for action, values
                in ranking[
                    "action_audit"
                ].items()
            },
            blocked_actions=tuple(
                ranking["blocked_actions"]
            ),
            trajectory_decision_id=(
                trajectory_decision_id
            ),
            selected_trajectory_id=(
                selected_trajectory_id
            ),
            planned_actions=tuple(
                planned_actions
            ),
            trajectory_failure_bounds=(
                trajectory_failure_bounds
            ),
            candidate_action_instances={
                reference: identity.instance_id
                for reference, identity
                in identities.items()
            },
            selected_action_family=(
                selected_identity.family
            ),
            selected_action_instance_id=(
                selected_identity.instance_id
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            objective_profile_instance_id=(
                self.objective_profile.instance_id
            ),
            objective_profile_signature=(
                self.objective_profile.signature
            ),
        )

        # Keep versioned planned action provenance in uncertainty audit rather
        # than changing the V3.2 public planned_actions tuple.
        if record.uncertainty_audit is not None:
            record.uncertainty_audit[
                "__trajectory_provenance__"
            ] = {
                "planned_action_instance_ids":
                    planned_instance_ids,
            }

        self.decision_memory.append(record)

        return {
            "decision": record,
            "ranking": ranking,
            "base_policy_scores": base_scores,
            "utility_estimates": utilities,
            "epistemic_scores":
                normalized_epistemic,
            "exploration_scores":
                explorations,
        }

    def choose_action_uncertainty_aware(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[Dict[str, float]] = None,
        risk_profile: Optional[UncertaintyRiskProfile] = None,
        belief_context_id: Optional[str] = None,
        *,
        action_estimates: Optional[Dict[str, ActionRiskEstimate]] = None,
        model_reliability: float = 1.0,
        requested_mode: Optional[RiskMode] = None,
        trajectory_decision_id: Optional[int] = None,
        selected_trajectory_id: Optional[str] = None,
        planned_actions: Tuple[str, ...] = (),
        trajectory_failure_bounds: Optional[Tuple[float, float]] = None,
    ):
        """Compatibility façade for canonical V2.21 and V3.1/V3.2 callers.

        A third positional mapping of ActionRiskEstimate is recognized as the
        V3 contract. Numeric epistemic mappings retain the V2.21 contract.
        """
        if action_estimates is None and epistemic_scores:
            if all(
                isinstance(value, ActionRiskEstimate)
                for value in epistemic_scores.values()
            ):
                action_estimates = dict(epistemic_scores)
                epistemic_scores = None

        if action_estimates is None:
            if requested_mode is not None or trajectory_decision_id is not None:
                raise ValueError(
                    "V3 risk/provenance arguments require action_estimates"
                )
            return self._choose_action_uncertainty_aware_v221(
                context=context,
                candidates=candidates,
                epistemic_scores=epistemic_scores,
                risk_profile=risk_profile,
                belief_context_id=belief_context_id,
            )

        if risk_profile is not None:
            raise ValueError(
                "risk_profile V2.21 tidak boleh dicampur dengan action_estimates V3"
            )
        return self._choose_action_uncertainty_aware_v32(
            context=context,
            candidates=candidates,
            action_estimates=action_estimates,
            epistemic_scores=epistemic_scores,
            belief_context_id=belief_context_id,
            model_reliability=model_reliability,
            requested_mode=requested_mode,
            trajectory_decision_id=trajectory_decision_id,
            selected_trajectory_id=selected_trajectory_id,
            planned_actions=planned_actions,
            trajectory_failure_bounds=trajectory_failure_bounds,
        )

    def record_transition_outcome(
        self,
        decision_id: int,
        reward: Optional[float],
        next_context: Optional[str],
        next_actions: Optional[List[str]],
        done: bool,
        next_belief_context_id: Optional[str] = None,
        next_action_instance_map: Optional[
            Dict[str, str]
        ] = None,
        next_state_key: Optional[str] = None,
        objective_outcome=None,
    ) -> Dict:
        """
        Sequential TD learning attributed to the exact action instance that
        produced the decision. Successor Q lookups use active successor action
        instances, preventing v1 experience from bootstrapping through v2
        under the same logical name.
        """
        record = self.decision_memory.get(
            decision_id
        )
        if record is None:
            raise KeyError(
                f"Decision {decision_id} tidak ditemukan"
            )
        if record.reward is not None:
            raise ValueError(
                f"Decision {decision_id} sudah memiliki reward"
            )

        action_instance = (
            record.selected_action_instance_id
            if record.selected_action_instance_id
            is not None
            else record.selected_action
        )
        action_family = (
            record.selected_action_family
            if record.selected_action_family
            is not None
            else record.selected_action
        )

        current_state_key = (
            record.state_key
            if record.state_key is not None
            else self.state_learning_key(
                record.context
            )
        )
        profile_reference = (
            record.objective_profile_instance_id
            if record.objective_profile_instance_id
                is not None
            else self._objective_compatibility_instance_id
        )
        current_scalar_state_key = (
            record.scalar_state_key
            if record.scalar_state_key is not None
            else self._objective_scalar_state_key_from_canonical(
                current_state_key,
                objective_profile_reference=(
                    profile_reference
                ),
            )
        )

        (
            resolved_reward,
            structured_outcome,
            objective_aggregation,
        ) = self._resolve_actual_utility(
            reward,
            objective_outcome,
            objective_profile_reference=(
                profile_reference
            ),
        )

        before = self.decision_policy.utility(
            current_scalar_state_key,
            action_instance,
            belief_context_id=(
                record.belief_context_id
            ),
        )

        resolved_next_belief_context = (
            record.belief_context_id
            if next_belief_context_id is None
            else next_belief_context_id
        )

        next_refs = sorted(
            set(next_actions or [])
        )
        if next_action_instance_map is not None:
            if set(
                next_action_instance_map
            ) != set(next_refs):
                raise ValueError(
                    "next_action_instance_map harus tepat "
                    "mencakup next_actions"
                )
            next_instance_ids = []
            for ref in next_refs:
                exact_reference = (
                    next_action_instance_map[
                        ref
                    ]
                )
                identity = (
                    self.resolve_action_identity(
                        exact_reference,
                        require_active=False,
                    )
                )
                next_instance_ids.append(
                    identity.instance_id
                )
        else:
            next_identities = (
                self._resolve_action_candidates(
                    next_refs
                )
                if next_refs
                else {}
            )
            next_instance_ids = [
                next_identities[
                    ref
                ].instance_id
                for ref in next_refs
            ]

        resolved_next_state_key = (
            next_state_key
            if next_state_key is not None
            else (
                self.state_learning_key(
                    next_context
                )
                if next_context is not None
                else None
            )
        )
        resolved_next_scalar_state_key = (
            None
            if resolved_next_state_key is None
            else self._objective_scalar_state_key_from_canonical(
                resolved_next_state_key,
                objective_profile_reference=(
                    profile_reference
                ),
            )
        )

        after = self.decision_policy.update_transition(
            current_scalar_state_key,
            action_instance,
            resolved_reward,
            resolved_next_scalar_state_key,
            next_instance_ids,
            done,
            belief_context_id=(
                record.belief_context_id
            ),
            next_belief_context_id=(
                resolved_next_belief_context
            ),
        )

        record.reward = resolved_reward

        if structured_outcome is not None:
            record.objective_outcome = (
                structured_outcome.as_dict()
            )
            record.objective_aggregation = (
                objective_aggregation.as_dict()
            )
            record.objective_profile_signature = (
                objective_aggregation.profile_signature
            )

        self._transition_counter += 1
        transition = TransitionRecord(
            transition_id=self._transition_counter,
            decision_id=decision_id,
            context=record.context,
            action=record.selected_action,
            reward=resolved_reward,
            next_context=next_context,
            next_actions=tuple(next_refs),
            done=done,
            utility_before=before,
            utility_after=after,
            belief_context_id=(
                record.belief_context_id
            ),
            next_belief_context_id=(
                resolved_next_belief_context
            ),
            action_family=action_family,
            action_instance_id=action_instance,
            next_action_instance_ids=tuple(
                next_instance_ids
            ),
            state_key=current_state_key,
            next_state_key=resolved_next_state_key,
            scalar_state_key=(
                current_scalar_state_key
            ),
            next_scalar_state_key=(
                resolved_next_scalar_state_key
            ),
            objective_profile_instance_id=(
                profile_reference
            ),
            objective_outcome=(
                None
                if structured_outcome is None
                else structured_outcome.as_dict()
            ),
            objective_aggregation=(
                None
                if objective_aggregation is None
                else objective_aggregation.as_dict()
            ),
            objective_profile_signature=(
                None
                if objective_aggregation is None
                else objective_aggregation.profile_signature
            ),
        )
        self.transition_memory.append(
            transition
        )
        self.maintain_memory(
            ("transition", "decision")
        )

        return {
            "transition_id":
                transition.transition_id,
            "decision_id": decision_id,
            "context": record.context,
            "belief_context_id":
                record.belief_context_id,
            "action": record.selected_action,
            "action_family": action_family,
            "action_instance_id":
                action_instance,
            "state_key": current_state_key,
            "scalar_state_key":
                current_scalar_state_key,
            "objective_profile_instance_id":
                profile_reference,
            "reward": resolved_reward,
            "objective_outcome": (
                None
                if structured_outcome is None
                else structured_outcome.as_dict()
            ),
            "objective_aggregation": (
                None
                if objective_aggregation is None
                else objective_aggregation.as_dict()
            ),
            "next_context": next_context,
            "next_actions": tuple(next_refs),
            "next_action_instance_ids":
                tuple(next_instance_ids),
            "next_state_key":
                resolved_next_state_key,
            "next_scalar_state_key":
                resolved_next_scalar_state_key,
            "next_belief_context_id":
                resolved_next_belief_context,
            "done": done,
            "utility_before": before,
            "utility_after": after,
        }

    def record_decision_outcome(
        self,
        decision_id: int,
        reward: Optional[float] = None,
        objective_outcome=None,
    ) -> Dict:
        """
        One-step utility update attributed to the exact immutable action
        instance stored by the decision.
        """
        record = self.decision_memory.get(
            decision_id
        )
        if record is None:
            raise KeyError(
                f"Decision {decision_id} tidak ditemukan"
            )
        if record.reward is not None:
            raise ValueError(
                f"Decision {decision_id} sudah memiliki reward"
            )

        action_instance = (
            record.selected_action_instance_id
            if record.selected_action_instance_id
            is not None
            else record.selected_action
        )
        action_family = (
            record.selected_action_family
            if record.selected_action_family
            is not None
            else record.selected_action
        )

        current_state_key = (
            record.state_key
            if record.state_key is not None
            else self.state_learning_key(
                record.context
            )
        )
        profile_reference = (
            record.objective_profile_instance_id
            if record.objective_profile_instance_id
                is not None
            else self._objective_compatibility_instance_id
        )
        current_scalar_state_key = (
            record.scalar_state_key
            if record.scalar_state_key is not None
            else self._objective_scalar_state_key_from_canonical(
                current_state_key,
                objective_profile_reference=(
                    profile_reference
                ),
            )
        )
        (
            resolved_reward,
            structured_outcome,
            objective_aggregation,
        ) = self._resolve_actual_utility(
            reward,
            objective_outcome,
            objective_profile_reference=(
                profile_reference
            ),
        )

        new_utility = self.decision_policy.update(
            current_scalar_state_key,
            action_instance,
            resolved_reward,
            belief_context_id=(
                record.belief_context_id
            ),
        )
        record.reward = resolved_reward

        if structured_outcome is not None:
            record.objective_outcome = (
                structured_outcome.as_dict()
            )
            record.objective_aggregation = (
                objective_aggregation.as_dict()
            )
            record.objective_profile_signature = (
                objective_aggregation.profile_signature
            )
        self.maintain_memory(
            ("decision",)
        )

        return {
            "decision_id": decision_id,
            "context": record.context,
            "belief_context_id":
                record.belief_context_id,
            "action": record.selected_action,
            "action_family": action_family,
            "action_instance_id":
                action_instance,
            "state_key":
                current_state_key,
            "scalar_state_key":
                current_scalar_state_key,
            "objective_profile_instance_id":
                profile_reference,
            "reward": resolved_reward,
            "objective_outcome": (
                None
                if structured_outcome is None
                else structured_outcome.as_dict()
            ),
            "objective_aggregation": (
                None
                if objective_aggregation is None
                else objective_aggregation.as_dict()
            ),
            "new_utility": new_utility,
            "count": self.decision_policy.count(
                current_scalar_state_key,
                action_instance,
                belief_context_id=(
                    record.belief_context_id
                ),
            ),
        }

    def record_objective_experience(
        self,
        decision_id: int,
        objective_outcome,
        success: bool,
        prediction_id: Optional[int] = None,
        next_context: Optional[str] = None,
        next_actions: Optional[List[str]] = None,
        done: Optional[bool] = None,
        next_belief_context_id: Optional[str] = None,
        next_action_instance_map: Optional[
            Dict[str, str]
        ] = None,
        next_state_key: Optional[str] = None,
    ) -> Dict:
        """
        Integrated ACTUAL-experience path for structured objectives.

        One ObjectiveOutcome is aggregated once and then reused for:
        - one-step Q update OR TD transition update;
        - scalar empirical world model;
        - vector objective world model;
        - selected forecast calibration when available.

        This avoids accidental subsystem disagreement about the same actual
        experience.

        If done is None -> one-step decision outcome.
        If done is bool -> sequential transition outcome.
        """
        record = self.decision_memory.get(
            decision_id
        )
        if record is None:
            raise KeyError(
                f"Decision {decision_id} tidak ditemukan"
            )
        if record.reward is not None:
            raise ValueError(
                f"Decision {decision_id} sudah memiliki reward"
            )

        profile_reference = (
            record.objective_profile_instance_id
            if record.objective_profile_instance_id
                is not None
            else self._objective_compatibility_instance_id
        )

        structured = ObjectiveOutcome.coerce(
            objective_outcome
        )
        aggregation = (
            self.aggregate_objective_outcome(
                structured,
                objective_profile_reference=(
                    profile_reference
                ),
            )
        )
        scalar_reward = (
            aggregation.scalar_utility
        )

        selected_prediction = None
        resolved_prediction_id = prediction_id

        if resolved_prediction_id is None:
            if (
                record.prediction_ids
                is not None
                and record.selected_action
                    in record.prediction_ids
            ):
                resolved_prediction_id = (
                    record.prediction_ids[
                        record.selected_action
                    ]
                )

        if resolved_prediction_id is not None:
            selected_prediction = (
                self.prediction_memory.get(
                    resolved_prediction_id
                )
            )
            if selected_prediction is None:
                raise KeyError(
                    "Prediction "
                    f"{resolved_prediction_id} tidak ditemukan"
                )

            if (
                selected_prediction.belief_context_id
                != record.belief_context_id
                or selected_prediction.context
                    != record.context
            ):
                raise ValueError(
                    "Prediction tidak cocok dengan decision experience"
                )

            prediction_profile = (
                selected_prediction.objective_profile_instance_id
                if selected_prediction.objective_profile_instance_id
                    is not None
                else self._objective_compatibility_instance_id
            )
            if (
                prediction_profile
                != profile_reference
            ):
                raise ObjectiveProfileVersionConflict(
                    "Prediction objective profile tidak cocok dengan decision"
                )

            expected_instance = (
                record.selected_action_instance_id
            )
            if (
                expected_instance is not None
                and selected_prediction.action_instance_id
                    not in (
                        None,
                        expected_instance,
                    )
            ):
                raise ValueError(
                    "Prediction action version tidak cocok dengan decision"
                )

        if done is None:
            learning_result = (
                self.record_decision_outcome(
                    decision_id,
                    reward=scalar_reward,
                    objective_outcome=structured,
                )
            )
        else:
            learning_result = (
                self.record_transition_outcome(
                    decision_id,
                    reward=scalar_reward,
                    next_context=next_context,
                    next_actions=next_actions,
                    done=bool(done),
                    next_belief_context_id=(
                        next_belief_context_id
                    ),
                    next_action_instance_map=(
                        next_action_instance_map
                    ),
                    next_state_key=(
                        next_state_key
                    ),
                    objective_outcome=structured,
                )
            )

        world_result = (
            self.record_world_model_outcome(
                context=record.context,
                action_name=(
                    record.selected_action
                ),
                reward=scalar_reward,
                success=bool(success),
                belief_context_id=(
                    record.belief_context_id
                ),
                action_instance_id=(
                    record.selected_action_instance_id
                ),
                state_key=(
                    record.state_key
                ),
                objective_outcome=structured,
                objective_profile_reference=(
                    profile_reference
                ),
                archive_objective_experience=False,
            )
        )

        objective_experience = (
            self._archive_actual_objective_experience(
                context=record.context,
                belief_context_id=(
                    record.belief_context_id
                ),
                state_key=(
                    record.state_key
                    if record.state_key
                        is not None
                    else self.state_learning_key(
                        record.context
                    )
                ),
                action_name=(
                    record.selected_action
                ),
                action_family=(
                    record.selected_action_family
                    if record.selected_action_family
                        is not None
                    else record.selected_action
                ),
                action_instance_id=(
                    record.selected_action_instance_id
                    if record.selected_action_instance_id
                        is not None
                    else record.selected_action
                ),
                objective_outcome=(
                    structured
                ),
                source_event=(
                    "integrated_objective_experience"
                ),
                decision_id=(
                    decision_id
                ),
                transition_id=(
                    learning_result.get(
                        "transition_id"
                    )
                    if isinstance(
                        learning_result,
                        dict,
                    )
                    else None
                ),
                success=bool(
                    success
                ),
                scalarization_profile_instance_id=(
                    profile_reference
                ),
                derived_scalar_utility=(
                    scalar_reward
                ),
            )
        )

        prediction_error = None
        if selected_prediction is not None:
            prediction_error = (
                self.assess_outcome_prediction(
                    selected_prediction,
                    actual_reward=scalar_reward,
                    actual_success=bool(success),
                    actual_objective_outcome=(
                        structured
                    ),
                )
            )

        return {
            "decision_id": decision_id,
            "objective_outcome":
                structured.as_dict(),
            "objective_aggregation":
                aggregation.as_dict(),
            "reward": scalar_reward,
            "success": bool(success),
            "learning":
                learning_result,
            "world_model":
                world_result,
            "prediction_error":
                prediction_error,
            "prediction_id":
                resolved_prediction_id,
            "objective_profile_instance_id":
                profile_reference,
            "objective_experience_id":
                objective_experience.experience_id,
        }


    # =====================================================================
    # V2.29 — preference-aware risk / trajectory integration
    # =====================================================================

    @staticmethod
    def _bounded_mean_interval(
        mean: float,
        support: int,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Bounded Hoeffding sidecar; explicitly not scalar calibration."""
        if support <= 0:
            return (0.0, 1.0, 0.5)
        alpha = 1.0 - confidence_level
        radius = math.sqrt(
            math.log(2.0 / alpha)
            / (2.0 * support)
        )
        radius = min(1.0, radius)
        return (
            max(0.0, mean - radius),
            min(1.0, mean + radius),
            radius,
        )

    def preference_aware_utility_estimate(
        self,
        context: str,
        action_reference: str,
        objective_profile_reference: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> PreferenceAwareUtilityEstimate:
        """Build a read-only utility estimate with exact preference provenance.

        Priority:
        1. actual Q samples under the exact requested/current profile;
        2. exact joint actual-vector reweighting;
        3. neutral prior.
        """
        profile = self.resolve_objective_profile(objective_profile_reference)
        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(context)
        identity = self.resolve_action_identity(
            action_reference,
            require_active=False,
        )
        scalar_state_key = self._objective_scalar_state_key_from_canonical(
            state_key,
            objective_profile_reference=profile.instance_id,
        )
        q_count = self.decision_policy.count(
            scalar_state_key,
            identity.instance_id,
            belief_context_id=scope,
        )

        if q_count > 0:
            mean = self.decision_policy.utility(
                scalar_state_key,
                identity.instance_id,
                belief_context_id=scope,
            )
            lower, upper, radius = self._bounded_mean_interval(mean, q_count)
            return PreferenceAwareUtilityEstimate(
                belief_context_id=scope,
                context=context,
                state_key=state_key,
                action_reference=action_reference,
                action_instance_id=identity.instance_id,
                objective_profile_instance_id=profile.instance_id,
                objective_profile_signature=profile.signature,
                source="profile_q",
                mean=mean,
                variance=None,
                aleatoric_std=None,
                epistemic_lower=lower,
                epistemic_upper=upper,
                epistemic_radius=radius,
                support=q_count,
                total_count=q_count,
                unscorable_count=0,
                coverage=1.0,
                mask_count=0,
                q_sample_count=q_count,
                learning_mutation=False,
                reweighted_history_is_scalar_calibration=False,
            )

        distribution = self.joint_objective_model.reweighted_distribution(
            state_key,
            identity.instance_id,
            profile,
            belief_context_id=scope,
        )
        support = int(distribution["scorable_count"])
        if support > 0 and distribution["mean"] is not None:
            mean = float(distribution["mean"])
            lower, upper, radius = self._bounded_mean_interval(mean, support)
            return PreferenceAwareUtilityEstimate(
                belief_context_id=scope,
                context=context,
                state_key=state_key,
                action_reference=action_reference,
                action_instance_id=identity.instance_id,
                objective_profile_instance_id=profile.instance_id,
                objective_profile_signature=profile.signature,
                source="reweighted_actual_vector_history",
                mean=mean,
                variance=distribution["variance"],
                aleatoric_std=distribution["std"],
                epistemic_lower=lower,
                epistemic_upper=upper,
                epistemic_radius=radius,
                support=support,
                total_count=int(distribution["total_count"]),
                unscorable_count=int(distribution["unscorable_count"]),
                coverage=float(distribution["coverage"]),
                mask_count=int(distribution["mask_count"]),
                q_sample_count=0,
                learning_mutation=False,
                reweighted_history_is_scalar_calibration=False,
            )

        mean = self.decision_policy.initial_utility
        return PreferenceAwareUtilityEstimate(
            belief_context_id=scope,
            context=context,
            state_key=state_key,
            action_reference=action_reference,
            action_instance_id=identity.instance_id,
            objective_profile_instance_id=profile.instance_id,
            objective_profile_signature=profile.signature,
            source="neutral_prior",
            mean=mean,
            variance=None,
            aleatoric_std=None,
            epistemic_lower=0.0,
            epistemic_upper=1.0,
            epistemic_radius=0.5,
            support=0,
            total_count=int(distribution["total_count"]),
            unscorable_count=int(distribution["unscorable_count"]),
            coverage=float(distribution["coverage"]),
            mask_count=int(distribution["mask_count"]),
            q_sample_count=0,
            learning_mutation=False,
            reweighted_history_is_scalar_calibration=False,
        )

    def validate_preference_aware_utility_estimate(
        self,
        estimate: PreferenceAwareUtilityEstimate,
        context: Optional[str] = None,
        action_reference: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        objective_profile_reference: Optional[str] = None,
    ) -> PreferenceAwareUtilityEstimate:
        profile = self.resolve_objective_profile(objective_profile_reference)
        if estimate.objective_profile_instance_id != profile.instance_id:
            raise ObjectiveProfileVersionConflict(
                "Stale/mismatched preference-aware objective profile: "
                f"{estimate.objective_profile_instance_id} != {profile.instance_id}"
            )
        if estimate.objective_profile_signature != profile.signature:
            raise ObjectiveProfileVersionConflict(
                "Preference-aware objective profile signature tidak cocok"
            )

        expected_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        if estimate.belief_context_id != expected_scope:
            raise ValueError(
                "Stale/mismatched preference-aware BeliefContext: "
                f"{estimate.belief_context_id} != {expected_scope}"
            )

        if context is not None:
            expected_state = self.state_learning_key(context)
            if estimate.state_key != expected_state:
                raise StateIdentityConflict(
                    "Stale/mismatched preference-aware state: "
                    f"{estimate.state_key} != {expected_state}"
                )

        resolved_reference = (
            action_reference
            if action_reference is not None
            else estimate.action_reference
        )
        identity = self.resolve_action_identity(
            resolved_reference,
            require_active=False,
        )
        if estimate.action_instance_id != identity.instance_id:
            raise ActionVersionConflict(
                "Stale/mismatched preference-aware action version: "
                f"{estimate.action_instance_id} != {identity.instance_id}"
            )
        if action_reference is not None and estimate.action_reference != action_reference:
            raise ValueError("Preference-aware action reference tidak cocok")
        if estimate.learning_mutation:
            raise ValueError("Preference-aware estimate tidak boleh mutate learning")
        return estimate

    def _preference_technical_estimate(
        self,
        context: str,
        action_reference: str,
        failure_consequence: float,
        belief_context_id: Optional[str] = None,
    ) -> ActionRiskEstimate:
        if not 0.0 <= failure_consequence <= 1.0:
            raise ValueError("failure consequence harus di [0,1]")
        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(context)
        identity = self.resolve_action_identity(
            action_reference,
            require_active=False,
        )
        stats = self.success_constraint_model.statistics(
            state_key,
            identity.instance_id,
            scope,
        )
        probability = float(stats["success_probability"])
        count = int(stats["count"])
        uncertainty = self.prediction_uncertainty_estimator.estimate(
            reward_mean=0.5,
            reward_std=None,
            success_probability=probability,
            sample_count=count,
            success_sample_count=count,
        )
        aleatoric = math.sqrt(max(0.0, probability * (1.0 - probability)))
        return ActionRiskEstimate(
            action=action_reference,
            predicted_reward=0.5,
            success_probability=probability,
            success_lower=uncertainty.success_lower,
            success_upper=uncertainty.success_upper,
            epistemic_uncertainty=uncertainty.success_epistemic_radius,
            aleatoric_uncertainty=aleatoric,
            state_uncertainty=(1.0 / math.sqrt(count + 1.0)),
            sample_count=count,
            failure_consequence=failure_consequence,
            information_gain=(1.0 / math.sqrt(count + 1.0)),
            action_instance_id=identity.instance_id,
            state_key=state_key,
            objective_profile_instance_id=self.objective_profile.instance_id,
            belief_context_id=scope,
        )

    def preference_aware_risk_state(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[Dict[str, float]] = None,
        failure_consequences: Optional[Dict[str, float]] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        identities = self._resolve_action_candidates(candidates)
        references = list(identities)
        scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(context)
        scalar_state_key = self.objective_scalar_state_key(context)

        epistemic_scores = epistemic_scores or {}
        unknown = set(epistemic_scores) - set(references)
        if unknown:
            raise ValueError(
                f"epistemic_scores memiliki action bukan candidate: {sorted(unknown)}"
            )
        consequences = {reference: 0.5 for reference in references}
        if failure_consequences is not None:
            unknown = set(failure_consequences) - set(references)
            if unknown:
                raise ValueError(
                    f"failure_consequences memiliki action bukan candidate: {sorted(unknown)}"
                )
            for action, value in failure_consequences.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError("failure consequence harus di [0,1]")
                consequences[action] = value

        utility_estimates = {
            reference: self.preference_aware_utility_estimate(
                context,
                reference,
                belief_context_id=scope,
            )
            for reference in references
        }
        technical_estimates = {
            reference: self._preference_technical_estimate(
                context,
                reference,
                consequences[reference],
                belief_context_id=scope,
            )
            for reference in references
        }

        normalized_epistemic = {
            reference: max(0.0, min(1.0, epistemic_scores.get(reference, 0.0)))
            for reference in references
        }
        exploration_scores = {
            reference: self.decision_policy.exploration_bonus(
                scalar_state_key,
                identities[reference].instance_id,
                len(references),
                belief_context_id=scope,
            )
            for reference in references
        }
        nonutility_scores = {
            reference: (
                self.decision_policy.epistemic_weight * normalized_epistemic[reference]
                + self.decision_policy.exploration_weight * exploration_scores[reference]
            )
            for reference in references
        }
        return {
            "belief_context_id": scope,
            "context": context,
            "state_key": state_key,
            "scalar_state_key": scalar_state_key,
            "objective_profile_instance_id": self.objective_profile.instance_id,
            "objective_profile_signature": self.objective_profile.signature,
            "identities": identities,
            "utility_estimates": utility_estimates,
            "technical_estimates": technical_estimates,
            "epistemic_scores": normalized_epistemic,
            "exploration_scores": exploration_scores,
            "nonutility_scores": nonutility_scores,
            "failure_consequences": consequences,
            "learning_mutation": False,
        }

    def choose_action_preference_aware_risk(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[Dict[str, float]] = None,
        failure_consequences: Optional[Dict[str, float]] = None,
        belief_context_id: Optional[str] = None,
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        self.maintain_memory(("decision", "transition"))
        bundle = self.preference_aware_risk_state(
            context,
            candidates,
            epistemic_scores=epistemic_scores,
            failure_consequences=failure_consequences,
            belief_context_id=belief_context_id,
        )
        ranking = self.preference_aware_risk_policy.rank(
            nonutility_scores=bundle["nonutility_scores"],
            utility_estimates=bundle["utility_estimates"],
            technical_estimates=bundle["technical_estimates"],
            requested_mode=requested_mode,
        )
        selected = ranking["selected_action"]
        if selected is None:
            return {"decision": None, "ranking": ranking, "state": bundle}

        identity = bundle["identities"][selected]
        self._decision_counter += 1
        record = DecisionRecord(
            decision_id=self._decision_counter,
            context=context,
            candidates=tuple(bundle["identities"]),
            selected_action=selected,
            policy_scores=dict(bundle["nonutility_scores"]),
            utility_estimates={
                action: estimate.mean
                for action, estimate in bundle["utility_estimates"].items()
            },
            epistemic_scores=dict(bundle["epistemic_scores"]),
            exploration_scores=dict(bundle["exploration_scores"]),
            belief_context_id=bundle["belief_context_id"],
            selection_mode="preference_aware_risk",
            strategy_scores=dict(ranking["risk_adjusted_scores"]),
            risk_mode=ranking["risk_mode"].value,
            uncertainty_scores={
                action: estimate.epistemic_radius
                for action, estimate in bundle["utility_estimates"].items()
            },
            blocked_actions=tuple(ranking["blocked_actions"]),
            risk_adjusted_scores=dict(ranking["risk_adjusted_scores"]),
            uncertainty_audit={
                action: dict(values)
                for action, values in ranking["action_audit"].items()
            },
            candidate_action_instances={
                reference: item.instance_id
                for reference, item in bundle["identities"].items()
            },
            selected_action_family=identity.family,
            selected_action_instance_id=identity.instance_id,
            state_key=bundle["state_key"],
            scalar_state_key=bundle["scalar_state_key"],
            objective_profile_instance_id=self.objective_profile.instance_id,
            objective_profile_signature=self.objective_profile.signature,
        )
        self.decision_memory.append(record)
        self.maintain_memory(("decision",))
        return {"decision": record, "ranking": ranking, "state": bundle}

    def build_preference_aware_trajectory_estimate(
        self,
        trajectory_id: str,
        steps: Tuple[Tuple[str, str], ...],
        failure_consequences: Optional[Tuple[float, ...]] = None,
        belief_context_id: Optional[str] = None,
    ) -> PreferenceAwareTrajectoryEstimate:
        normalized = tuple(steps)
        if not normalized:
            raise ValueError("Trajectory preference-aware tidak boleh kosong")
        if failure_consequences is None:
            consequences = tuple(0.5 for _ in normalized)
        else:
            consequences = tuple(failure_consequences)
            if len(consequences) != len(normalized):
                raise ValueError("failure_consequences trajectory harus sama panjang")
            if any(not 0.0 <= value <= 1.0 for value in consequences):
                raise ValueError("trajectory failure consequence harus di [0,1]")

        utility_steps = []
        technical_steps = []
        for (context, action_reference), consequence in zip(normalized, consequences):
            utility_steps.append(
                self.preference_aware_utility_estimate(
                    context,
                    action_reference,
                    belief_context_id=belief_context_id,
                )
            )
            technical_steps.append(
                self._preference_technical_estimate(
                    context,
                    action_reference,
                    consequence,
                    belief_context_id=belief_context_id,
                )
            )
        return PreferenceAwareTrajectoryEstimate(
            trajectory_id=trajectory_id,
            utility_steps=tuple(utility_steps),
            technical_steps=tuple(technical_steps),
        )

    def validate_preference_aware_trajectory_estimate(
        self,
        estimate: PreferenceAwareTrajectoryEstimate,
        steps: Optional[Tuple[Tuple[str, str], ...]] = None,
        belief_context_id: Optional[str] = None,
    ) -> PreferenceAwareTrajectoryEstimate:
        if estimate.objective_profile_instance_id != self.objective_profile.instance_id:
            raise ObjectiveProfileVersionConflict(
                "Stale/mismatched preference-aware trajectory profile"
            )
        if estimate.objective_profile_signature != self.objective_profile.signature:
            raise ObjectiveProfileVersionConflict(
                "Preference-aware trajectory profile signature tidak cocok"
            )
        expected_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        if estimate.belief_context_id != expected_scope:
            raise ValueError("Preference-aware trajectory BeliefContext tidak cocok")

        if steps is not None:
            normalized = tuple(steps)
            if len(normalized) != estimate.horizon:
                raise ValueError("Trajectory steps tidak sama panjang dengan estimate")
            for expected, utility in zip(normalized, estimate.utility_steps):
                context, action_reference = expected
                self.validate_preference_aware_utility_estimate(
                    utility,
                    context=context,
                    action_reference=action_reference,
                    belief_context_id=expected_scope,
                )
        return estimate

    def choose_trajectory_preference_aware_risk(
        self,
        candidate_trajectories: Dict[str, Tuple[Tuple[str, str], ...]],
        requested_mode: Optional[RiskMode] = None,
        failure_consequences: Optional[Dict[str, Tuple[float, ...]]] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        if not candidate_trajectories:
            raise ValueError("candidate_trajectories tidak boleh kosong")
        estimates = {}
        for trajectory_id, steps in candidate_trajectories.items():
            estimates[trajectory_id] = self.build_preference_aware_trajectory_estimate(
                trajectory_id,
                tuple(steps),
                failure_consequences=(
                    None
                    if failure_consequences is None
                    else failure_consequences.get(trajectory_id)
                ),
                belief_context_id=belief_context_id,
            )
        ranking = self.preference_aware_trajectory_risk_policy.rank(
            estimates,
            requested_mode=requested_mode,
        )
        return {
            "decision": None,
            "ranking": ranking,
            "trajectory_estimates": estimates,
            "counterfactual_is_experience": False,
            "learning_mutation": False,
        }


    def decision_state(
        self,
        context: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        """
        Default view = current belief context.

        Legacy unscoped stores remain exposed separately for compatibility.
        """
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )

        state_key = (
            self.state_learning_key(
                context
            )
            if context is not None
            else None
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
            if context is not None
            else None
        )
        scoped = self.decision_policy.scoped_state(
            resolved_scope,
            context=scalar_state_key,
        )

        decisions = [
            record
            for record in self.decision_memory.for_belief_context(
                resolved_scope
            )
            if (
                state_key is None
                or (
                    record.state_key
                    if record.state_key is not None
                    else record.context
                ) == state_key
            )
        ]

        transitions = [
            t
            for t in self.transition_memory.all()
            if t.belief_context_id == resolved_scope
            and (
                state_key is None
                or (
                    t.state_key
                    if t.state_key is not None
                    else t.context
                ) == state_key
            )
        ]

        return {
            "belief_context_id": resolved_scope,
            "context": context,
            "state_key": state_key,
            "scalar_state_key":
                scalar_state_key,
            "objective_profile":
                self.objective_profile_state(),
            "q_values": scoped["q_values"],
            "counts": scoped["counts"],
            "decisions": len(decisions),
            "transitions": len(transitions),
            "legacy_q_values": dict(
                self.decision_policy.q_values
            ),
            "legacy_counts": dict(
                self.decision_policy.counts
            ),
            "action_registry": (
                self.action_registry_state()
            ),
            "state_registry": (
                self.state_registry_state()
            ),
        }



    def predict_outcome(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
        objective_profile_reference: Optional[
            str
        ] = None,
    ) -> OutcomePrediction:
        """
        Forecast from ACTUAL experience statistics only.

        Registered logical actions resolve to immutable action-instance
        world-model keys. Explicit old instance IDs remain queryable.
        """
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(
            context
        )
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )
        scalar_state_key = (
            self._objective_scalar_state_key_from_canonical(
                state_key,
                objective_profile_reference=(
                    profile.instance_id
                ),
            )
        )
        identity = self.resolve_action_identity(
            action_name,
            require_active=False,
        )
        model_action = identity.instance_id

        objective_stats = (
            self.objective_world_model.statistics(
                state_key,
                model_action,
                belief_context_id=resolved_scope,
            )
        )
        predicted_objectives = {
            component: values["mean"]
            for component, values
            in objective_stats.items()
            if values["count"] > 0
        }
        objective_sample_counts = {
            component: values["count"]
            for component, values
            in objective_stats.items()
            if values["count"] > 0
        }

        reweighted_distribution = (
            self.joint_objective_model
            .reweighted_distribution(
                state_key,
                model_action,
                profile,
                belief_context_id=resolved_scope,
            )
        )

        stats = (
            self.contextual_world_model.statistics(
                scalar_state_key,
                model_action,
                belief_context_id=resolved_scope,
            )
        )
        success_stats = (
            self.success_constraint_model.statistics(
                state_key,
                model_action,
                belief_context_id=resolved_scope,
            )
        )

        uncertainty = (
            self.prediction_uncertainty_estimator.estimate(
                reward_mean=stats[
                    "reward_mean"
                ],
                reward_std=stats[
                    "reward_std"
                ],
                success_probability=success_stats[
                    "success_probability"
                ],
                sample_count=stats["count"],
                success_sample_count=(
                    success_stats["count"]
                ),
            )
        )

        reweighted_uncertainty = None
        if (
            reweighted_distribution[
                "mean"
            ] is not None
        ):
            reweighted_uncertainty = (
                self.prediction_uncertainty_estimator
                .estimate(
                    reward_mean=(
                        reweighted_distribution[
                            "mean"
                        ]
                    ),
                    reward_std=(
                        reweighted_distribution[
                            "std"
                        ]
                    ),
                    success_probability=(
                        success_stats[
                            "success_probability"
                        ]
                    ),
                    sample_count=(
                        reweighted_distribution[
                            "scorable_count"
                        ]
                    ),
                    success_sample_count=(
                        success_stats[
                            "count"
                        ]
                    ),
                )
            )

        calibration_action = (
            identity.instance_id
            if identity.registered
            else None
        )
        model_reliability = (
            self.world_model_reliability.reliability(
                scalar_state_key,
                belief_context_id=(
                    resolved_scope
                ),
                action_instance_id=(
                    calibration_action
                ),
            )
        )

        self._outcome_prediction_counter += 1
        prediction = OutcomePrediction(
            prediction_id=(
                self._outcome_prediction_counter
            ),
            context=context,
            action_name=action_name,
            belief_context_id=resolved_scope,
            predicted_reward=stats[
                "reward_mean"
            ],
            predicted_success_probability=(
                success_stats[
                    "success_probability"
                ]
            ),
            sample_count=stats["count"],
            uncertainty=uncertainty,
            success_sample_count=(
                success_stats["count"]
            ),
            model_reliability=(
                model_reliability
            ),
            action_instance_id=(
                identity.instance_id
                if identity.registered
                else None
            ),
            state_key=state_key,
            scalar_state_key=(
                scalar_state_key
            ),
            predicted_objectives=(
                predicted_objectives
            ),
            objective_sample_counts=(
                objective_sample_counts
            ),
            objective_profile_signature=(
                profile.signature
            ),
            objective_profile_instance_id=(
                profile.instance_id
            ),
            reweighted_objective_utility=(
                reweighted_distribution[
                    "mean"
                ]
            ),
            reweighted_objective_support=(
                reweighted_distribution[
                    "scorable_count"
                ]
            ),
            reweighted_objective_variance=(
                reweighted_distribution[
                    "variance"
                ]
            ),
            reweighted_objective_std=(
                reweighted_distribution[
                    "std"
                ]
            ),
            reweighted_objective_coverage=(
                reweighted_distribution[
                    "coverage"
                ]
            ),
            reweighted_objective_unscorable_count=(
                reweighted_distribution[
                    "unscorable_count"
                ]
            ),
            reweighted_objective_mask_count=(
                reweighted_distribution[
                    "mask_count"
                ]
            ),
            reweighted_objective_uncertainty=(
                reweighted_uncertainty
            ),
        )
        self.prediction_memory.append(
            prediction
        )
        self.maintain_memory(
            ("prediction",)
        )
        return prediction

    def inspect_prediction_uncertainty(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
        objective_profile_reference: Optional[
            str
        ] = None,
    ) -> Dict:
        """Read-only exact-version uncertainty inspection."""
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(
            context
        )
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )
        scalar_state_key = (
            self._objective_scalar_state_key_from_canonical(
                state_key,
                objective_profile_reference=(
                    profile.instance_id
                ),
            )
        )
        identity = self.resolve_action_identity(
            action_name,
            require_active=False,
        )
        model_action = identity.instance_id

        stats = (
            self.contextual_world_model.statistics(
                scalar_state_key,
                model_action,
                belief_context_id=resolved_scope,
            )
        )
        success_stats = (
            self.success_constraint_model.statistics(
                state_key,
                model_action,
                belief_context_id=resolved_scope,
            )
        )
        uncertainty = (
            self.prediction_uncertainty_estimator.estimate(
                reward_mean=stats[
                    "reward_mean"
                ],
                reward_std=stats[
                    "reward_std"
                ],
                success_probability=success_stats[
                    "success_probability"
                ],
                sample_count=stats["count"],
                success_sample_count=(
                    success_stats["count"]
                ),
            )
        )

        calibration_action = (
            identity.instance_id
            if identity.registered
            else None
        )
        return {
            "belief_context_id":
                resolved_scope,
            "context": context,
            "state_key": state_key,
            "scalar_state_key":
                scalar_state_key,
            "objective_profile_instance_id":
                profile.instance_id,
            "objective_profile_signature":
                profile.signature,
            "action_name": action_name,
            "action_family":
                identity.family,
            "action_instance_id":
                identity.instance_id,
            "registered_action":
                identity.registered,
            "prediction": {
                "reward_mean":
                    stats["reward_mean"],
                "success_probability":
                    success_stats[
                        "success_probability"
                    ],
                "sample_count":
                    stats["count"],
                "success_sample_count":
                    success_stats["count"],
            },
            "uncertainty": uncertainty,
            "model_reliability": (
                self.world_model_reliability.reliability(
                    scalar_state_key,
                    belief_context_id=(
                        resolved_scope
                    ),
                    action_instance_id=(
                        calibration_action
                    ),
                )
            ),
        }

    def record_world_model_outcome(
        self,
        context: str,
        action_name: str,
        reward: Optional[float],
        success: bool,
        belief_context_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
        state_key: Optional[str] = None,
        objective_outcome=None,
        objective_profile_reference: Optional[
            str
        ] = None,
        archive_objective_experience: bool = True,
    ) -> Dict:
        """
        Actual experience update for the empirical world model.

        action_instance_id can be supplied by a prior DecisionRecord to make
        delayed feedback safe across a later action supersession.
        """
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )

        resolved_state_key = (
            state_key
            if state_key is not None
            else self.state_learning_key(
                context
            )
        )
        profile = self.resolve_objective_profile(
            objective_profile_reference
        )
        resolved_scalar_state_key = (
            self._objective_scalar_state_key_from_canonical(
                resolved_state_key,
                objective_profile_reference=(
                    profile.instance_id
                ),
            )
        )

        reference = (
            action_instance_id
            if action_instance_id is not None
            else action_name
        )
        identity = self.resolve_action_identity(
            reference,
            require_active=False,
        )

        (
            resolved_reward,
            structured_outcome,
            objective_aggregation,
        ) = self._resolve_actual_utility(
            reward,
            objective_outcome,
            objective_profile_reference=(
                profile.instance_id
            ),
        )

        result = self.contextual_world_model.update(
            resolved_scalar_state_key,
            identity.instance_id,
            resolved_reward,
            success,
            belief_context_id=resolved_scope,
        )
        success_stats = (
            self.success_constraint_model.update(
                resolved_state_key,
                identity.instance_id,
                success,
                belief_context_id=resolved_scope,
            )
        )
        objective_stats = None
        joint_objective_stats = None
        if structured_outcome is not None:
            objective_stats = (
                self.objective_world_model.update(
                    resolved_state_key,
                    identity.instance_id,
                    structured_outcome,
                    belief_context_id=resolved_scope,
                )
            )
            joint_objective_stats = (
                self.joint_objective_model.update(
                    resolved_state_key,
                    identity.instance_id,
                    structured_outcome,
                    belief_context_id=resolved_scope,
                )
            )

        result["state_key"] = (
            resolved_state_key
        )
        result["scalar_state_key"] = (
            resolved_scalar_state_key
        )
        result[
            "objective_profile_instance_id"
        ] = profile.instance_id
        result[
            "objective_profile_signature"
        ] = profile.signature
        result["action_family"] = (
            identity.family
        )
        result["action_instance_id"] = (
            identity.instance_id
        )
        result["objective_outcome"] = (
            None
            if structured_outcome is None
            else structured_outcome.as_dict()
        )
        result["objective_aggregation"] = (
            None
            if objective_aggregation is None
            else objective_aggregation.as_dict()
        )
        result["objective_statistics"] = (
            objective_stats
        )
        result["joint_objective_statistics"] = (
            joint_objective_stats
        )
        result["success_constraint"] = (
            success_stats
        )

        objective_experience = None
        if (
            structured_outcome is not None
            and archive_objective_experience
        ):
            objective_experience = (
                self._archive_actual_objective_experience(
                    context=context,
                    belief_context_id=(
                        resolved_scope
                    ),
                    state_key=(
                        resolved_state_key
                    ),
                    action_name=(
                        action_name
                    ),
                    action_family=(
                        identity.family
                    ),
                    action_instance_id=(
                        identity.instance_id
                    ),
                    objective_outcome=(
                        structured_outcome
                    ),
                    source_event=(
                        "world_model_outcome"
                    ),
                    success=bool(
                        success
                    ),
                    scalarization_profile_instance_id=(
                        profile.instance_id
                    ),
                    derived_scalar_utility=(
                        resolved_reward
                    ),
                )
            )

        result[
            "objective_experience_id"
        ] = (
            None
            if objective_experience is None
            else objective_experience.experience_id
        )
        return result

    def assess_outcome_prediction(
        self,
        prediction: OutcomePrediction,
        actual_reward: Optional[float],
        actual_success: bool,
        actual_objective_outcome=None,
    ) -> PredictionErrorRecord:
        """
        Calibrate exactly the action version that produced the forecast.
        """
        profile_reference = (
            prediction.objective_profile_instance_id
            if prediction.objective_profile_instance_id
                is not None
            else self._objective_compatibility_instance_id
        )

        (
            resolved_actual_reward,
            structured_outcome,
            objective_aggregation,
        ) = self._resolve_actual_utility(
            actual_reward,
            actual_objective_outcome,
            objective_profile_reference=(
                profile_reference
            ),
        )

        reward_error = abs(
            prediction.predicted_reward
            - resolved_actual_reward
        )

        objective_errors = None
        if structured_outcome is not None:
            objective_errors = {}
            predicted_components = (
                prediction.predicted_objectives
                or {}
            )
            for (
                component,
                actual_value,
            ) in (
                structured_outcome
                .observed_components()
                .items()
            ):
                if component in predicted_components:
                    objective_errors[
                        component
                    ] = abs(
                        predicted_components[
                            component
                        ]
                        - actual_value
                    )
        success_error = (
            0.0
            if prediction.predicted_success
                == bool(actual_success)
            else 1.0
        )
        aggregate_error = (
            0.70 * reward_error
            + 0.30 * success_error
        )
        aggregate_error = max(
            0.0,
            min(1.0, aggregate_error),
        )
        accuracy = 1.0 - aggregate_error
        prediction_state_key = (
            prediction.state_key
            if prediction.state_key is not None
            else self.state_learning_key(
                prediction.context
            )
        )
        prediction_scalar_state_key = (
            prediction.scalar_state_key
            if prediction.scalar_state_key
                is not None
            else self._objective_scalar_state_key_from_canonical(
                prediction_state_key,
                objective_profile_reference=(
                    profile_reference
                ),
            )
        )

        calibration_action = (
            prediction.action_instance_id
        )
        before = (
            self.world_model_reliability.reliability(
                prediction_scalar_state_key,
                belief_context_id=(
                    prediction.belief_context_id
                ),
                action_instance_id=(
                    calibration_action
                ),
            )
        )
        after = (
            self.world_model_reliability.update(
                prediction_scalar_state_key,
                accuracy,
                belief_context_id=(
                    prediction.belief_context_id
                ),
                action_instance_id=(
                    calibration_action
                ),
            )
        )

        self._prediction_error_counter += 1
        record = PredictionErrorRecord(
            prediction_error_id=(
                self._prediction_error_counter
            ),
            context=prediction.context,
            action_name=prediction.action_name,
            predicted_reward=(
                prediction.predicted_reward
            ),
            actual_reward=resolved_actual_reward,
            reward_error=reward_error,
            safety_error=0.0,
            efficiency_error=0.0,
            success_error=success_error,
            aggregate_error=aggregate_error,
            model_accuracy=accuracy,
            prediction_world_signature=(
                "context_scoped_empirical",
                prediction.sample_count,
                prediction.success_sample_count,
                prediction.action_instance_id,
                profile_reference,
            ),
            actual_world_signature=(
                "actual_outcome",
                prediction.action_instance_id,
                profile_reference,
            ),
            state_drift=False,
            reliability_before=before,
            reliability_after=after,
            calibrated=True,
            belief_context_id=(
                prediction.belief_context_id
            ),
            action_instance_id=(
                prediction.action_instance_id
            ),
            state_key=prediction_state_key,
            scalar_state_key=(
                prediction_scalar_state_key
            ),
            objective_profile_instance_id=(
                profile_reference
            ),
            objective_errors=objective_errors,
            objective_profile_signature=(
                None
                if objective_aggregation is None
                else objective_aggregation.profile_signature
            ),
        )
        self.prediction_error_memory.append(
            record
        )

        self._active_prediction_pins.discard(
            prediction.prediction_id
        )

        self.maintain_memory(
            (
                "prediction_error",
                "prediction",
            )
        )
        return record

    def contextual_world_model_state(
        self,
        context: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = (
            self.state_learning_key(
                context
            )
            if context is not None
            else None
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
            if context is not None
            else None
        )
        model_state = self.contextual_world_model.state(
            belief_context_id=resolved_scope,
            context=scalar_state_key,
        )

        uncertainty_state = {}
        if context is not None:
            for action_name in model_state:
                uncertainty_state[action_name] = (
                    self.inspect_prediction_uncertainty(
                        context,
                        action_name,
                        belief_context_id=resolved_scope,
                    )["uncertainty"]
                )

        action_reliability = {}
        if context is not None:
            for action_name in model_state:
                identity = (
                    self.resolve_action_identity(
                        action_name,
                        require_active=False,
                    )
                )
                calibration_action = (
                    identity.instance_id
                    if identity.registered
                    else None
                )
                action_reliability[
                    action_name
                ] = (
                    self.world_model_reliability.state(
                        context=scalar_state_key,
                        belief_context_id=resolved_scope,
                        action_instance_id=(
                            calibration_action
                        ),
                    )
                    if calibration_action
                    is not None
                    else self.world_model_reliability.state(
                        context=scalar_state_key,
                        belief_context_id=resolved_scope,
                    )
                )

        return {
            "belief_context_id": resolved_scope,
            "context": context,
            "state_key": state_key,
            "scalar_state_key":
                scalar_state_key,
            "objective_profile":
                self.objective_profile_state(),
            "model": model_state,
            "uncertainty": uncertainty_state,
            "action_reliability": action_reliability,
            "predictions": len(
                self.prediction_memory.for_belief_context(
                    resolved_scope
                )
            ),
            "prediction_errors": len(
                self.prediction_error_memory.for_belief_context(
                    resolved_scope
                )
            ),
            "reliability": (
                self.world_model_reliability.state(
                    context=scalar_state_key,
                    belief_context_id=resolved_scope,
                )
                if context is not None
                else self.world_model_reliability.state(
                    belief_context_id=resolved_scope,
                )
            ),
            "objective_model": (
                self.objective_world_model.state(
                    belief_context_id=resolved_scope,
                    context=state_key,
                )
            ),
            "joint_objective_model": (
                self.joint_objective_model.state(
                    belief_context_id=resolved_scope,
                    context=state_key,
                )
            ),
            "joint_objective_group_count": (
                self.joint_objective_model.group_count(
                    belief_context_id=resolved_scope,
                    context=state_key,
                )
            ),
            "success_constraint_model": (
                self.success_constraint_model.state(
                    belief_context_id=resolved_scope,
                    context=state_key,
                )
            ),
            "objective_profile": (
                self.objective_profile_state()
            ),
        }


    def _world_signature(self) -> Tuple:
        """
        Fingerprint deterministic dari world model saat simulasi.
        Berguna untuk audit stale counterfactual.
        """
        static = tuple(sorted(
            (p.x, p.y)
            for p in self.costmap.static_obstacles
        ))
        moving = tuple(sorted(
            (
                obs.id,
                obs.is_looping,
                tuple((p.x, p.y) for p in obs.trajectory),
            )
            for obs in self.costmap.moving_obstacles
        ))
        return (
            self.costmap.width,
            self.costmap.height,
            static,
            moving,
        )

    def simulate_route_action(
        self,
        context: str,
        route_action: RouteAction,
        belief_context_id: Optional[str] = None,
    ) -> CounterfactualEstimate:
        """
        What-if simulation. The action instance is recorded for provenance,
        but simulation still does not update Q/world-model experience.
        """
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )
        state_key = self.state_learning_key(
            context
        )
        identity = self.resolve_action_identity(
            route_action.name,
            require_active=True,
        )

        signature = self._world_signature()
        trajectory, audits = (
            self.plan_and_interrogate_spacetime_route(
                route_action.start,
                route_action.goal,
            )
        )
        predicted = (
            self.world_outcome_evaluator.evaluate_route(
                route_action.start,
                route_action.goal,
                trajectory,
                audits,
            )
        )

        self._counterfactual_counter += 1
        estimate = CounterfactualEstimate(
            counterfactual_id=(
                self._counterfactual_counter
            ),
            context=context,
            action_name=route_action.name,
            world_signature=signature,
            trajectory=trajectory,
            audits=list(audits),
            predicted_outcome=predicted,
            belief_context_id=resolved_scope,
            action_instance_id=(
                identity.instance_id
                if identity.registered
                else None
            ),
            state_key=state_key,
        )
        self.counterfactual_memory.append(
            estimate
        )
        self.maintain_memory(
            ("counterfactual",)
        )
        return estimate

    def compare_route_strategies(
        self,
        context: str,
        route_actions: List[RouteAction],
        epistemic_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Bandingkan semua kandidat melalui model dunia, TANPA belajar dari
        hasil simulasi dan TANPA membuat keputusan aktual.
        """
        if not route_actions:
            raise ValueError("route_actions tidak boleh kosong")

        by_name: Dict[str, RouteAction] = {}
        for action in route_actions:
            if action.name in by_name:
                raise ValueError(
                    f"Nama route action duplikat: {action.name}"
                )
            by_name[action.name] = action

        candidates = sorted(by_name)

        state_key = self.state_learning_key(
            context
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
        )

        belief_context_id = (
            self.belief_contexts.current_id
        )

        (
            base_scores,
            utilities,
            epistemics,
            explorations,
        ) = self.decision_policy.score_actions(
            scalar_state_key,
            candidates,
            epistemic_scores,
            belief_context_id=belief_context_id,
        )

        estimates = {
            name: self.simulate_route_action(
                context,
                by_name[name],
                belief_context_id=belief_context_id,
            )
            for name in candidates
        }

        predicted_rewards = {
            name: estimates[name].predicted_outcome.reward
            for name in candidates
        }

        model_reliability = self.world_model_reliability.reliability(
            scalar_state_key,
            belief_context_id=belief_context_id,
        )
        effective_counterfactual_weight = (
            self.counterfactual_policy.effective_weight(
                model_reliability
            )
        )

        strategy_scores = self.counterfactual_policy.combine(
            base_scores,
            predicted_rewards,
            model_reliability=model_reliability,
        )

        selected = max(
            candidates,
            key=lambda action: (
                strategy_scores[action],
                action,
            ),
        )

        return {
            "context": context,
            "state_key": state_key,
            "scalar_state_key":
                scalar_state_key,
            "objective_profile_instance_id":
                self.objective_profile.instance_id,
            "belief_context_id": belief_context_id,
            "selected_action": selected,
            "base_policy_scores": dict(base_scores),
            "utility_estimates": dict(utilities),
            "epistemic_scores": dict(epistemics),
            "exploration_scores": dict(explorations),
            "predicted_rewards": dict(predicted_rewards),
            "strategy_scores": dict(strategy_scores),
            "model_reliability": model_reliability,
            "effective_counterfactual_weight": effective_counterfactual_weight,
            "counterfactuals": estimates,
        }


    def assess_prediction_error(
        self,
        context: str,
        prediction: CounterfactualEstimate,
        actual_outcome: WorldOutcome,
        actual_world_signature: Optional[Tuple] = None,
    ) -> PredictionErrorRecord:
        """
        Compare predicted vs actual route outcome and calibrate the exact
        action version when one was registered.
        """
        actual_state_key = self.state_learning_key(
            context
        )
        prediction_state_key = (
            prediction.state_key
            if prediction.state_key is not None
            else self.state_learning_key(prediction.context)
        )
        if prediction_state_key != actual_state_key:
            raise StateIdentityConflict(
                "Prediction state identity tidak cocok: "
                f"{prediction_state_key} != {actual_state_key}"
            )

        if actual_world_signature is None:
            actual_world_signature = (
                self._world_signature()
            )

        state_drift = (
            actual_world_signature
            != prediction.world_signature
        )

        errors = (
            self.prediction_error_evaluator.evaluate(
                prediction.predicted_outcome,
                actual_outcome,
            )
        )

        prediction_scope = (
            prediction.belief_context_id
            if prediction.belief_context_id
            is not None
            else self.belief_contexts.current_id
        )

        before = (
            self.world_model_reliability.reliability(
                prediction_state_key,
                belief_context_id=(
                    prediction_scope
                ),
                action_instance_id=(
                    prediction.action_instance_id
                ),
            )
        )

        calibrated = not state_drift
        if calibrated:
            after = (
                self.world_model_reliability.update(
                    prediction_state_key,
                    errors[
                        "model_accuracy"
                    ],
                    belief_context_id=(
                        prediction_scope
                    ),
                    action_instance_id=(
                        prediction.action_instance_id
                    ),
                )
            )
        else:
            after = before

        self._prediction_error_counter += 1
        record = PredictionErrorRecord(
            prediction_error_id=(
                self._prediction_error_counter
            ),
            context=context,
            action_name=prediction.action_name,
            predicted_reward=(
                prediction.predicted_outcome.reward
            ),
            actual_reward=(
                actual_outcome.reward
            ),
            reward_error=errors[
                "reward_error"
            ],
            safety_error=errors[
                "safety_error"
            ],
            efficiency_error=errors[
                "efficiency_error"
            ],
            success_error=errors[
                "success_error"
            ],
            aggregate_error=errors[
                "aggregate_error"
            ],
            model_accuracy=errors[
                "model_accuracy"
            ],
            prediction_world_signature=(
                prediction.world_signature
            ),
            actual_world_signature=(
                actual_world_signature
            ),
            state_drift=state_drift,
            reliability_before=before,
            reliability_after=after,
            calibrated=calibrated,
            belief_context_id=(
                prediction_scope
            ),
            action_instance_id=(
                prediction.action_instance_id
            ),
            state_key=prediction_state_key,
        )
        self.prediction_error_memory.append(
            record
        )
        return record

    def model_learning_state(
        self,
        context: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        resolved_scope = (
            belief_context_id
            if belief_context_id is not None
            else self.belief_contexts.current_id
        )

        state_key = (
            self.state_learning_key(
                context
            )
            if context is not None
            else None
        )
        scalar_state_key = (
            self.objective_scalar_state_key(
                context
            )
            if context is not None
            else None
        )

        if context is not None:
            records = [
                record
                for record in self.prediction_error_memory.for_belief_context(
                    resolved_scope
                )
                if (
                    record.state_key
                    if record.state_key is not None
                    else record.context
                ) == state_key
            ]
        else:
            records = (
                self.prediction_error_memory.for_belief_context(
                    resolved_scope
                )
            )

        calibrated = [
            r for r in records
            if r.calibrated
        ]

        avg_error = (
            sum(
                r.aggregate_error
                for r in calibrated
            ) / len(calibrated)
            if calibrated else 0.0
        )

        return {
            "belief_context_id": resolved_scope,
            "context": context,
            "state_key": state_key,
            "scalar_state_key":
                scalar_state_key,
            "objective_profile":
                self.objective_profile_state(),
            "reliability": (
                self.world_model_reliability.state(
                    scalar_state_key,
                    belief_context_id=resolved_scope,
                )
                if context is not None
                else self.world_model_reliability.state(
                    belief_context_id=resolved_scope,
                )
            ),
            "prediction_records": len(records),
            "forecast_records": len([
                record
                for record in self.prediction_memory.for_belief_context(
                    resolved_scope
                )
                if (
                    state_key is None
                    or (
                        record.state_key
                        if record.state_key is not None
                        else record.context
                    ) == state_key
                )
            ]),
            "calibrated_records": len(calibrated),
            "state_drift_records": sum(
                1 for r in records
                if r.state_drift
            ),
            "average_calibrated_error": avg_error,
            "empirical_world_model": (
                self.contextual_world_model.state(
                    belief_context_id=resolved_scope,
                    context=scalar_state_key,
                )
            ),
        }


    def choose_strategy_and_execute(
        self,
        context: str,
        route_actions: List[RouteAction],
        epistemic_scores: Optional[Dict[str, float]] = None,
    ) -> StrategyExecution:
        """
        1. Simulasikan semua alternatif.
        2. Pilih berdasarkan strategy score.
        3. Buat SATU decision aktual.
        4. Jalankan/plankan ulang action terpilih.
        5. Hanya outcome aktual yang mengubah Q-value.
        """
        comparison = self.compare_route_strategies(
            context,
            route_actions,
            epistemic_scores,
        )

        by_name = {
            action.name: action
            for action in route_actions
        }

        selected = comparison["selected_action"]

        self._decision_counter += 1
        decision = DecisionRecord(
            decision_id=self._decision_counter,
            context=context,
            candidates=tuple(sorted(by_name)),
            selected_action=selected,
            policy_scores=dict(
                comparison["base_policy_scores"]
            ),
            utility_estimates=dict(
                comparison["utility_estimates"]
            ),
            epistemic_scores=dict(
                comparison["epistemic_scores"]
            ),
            exploration_scores=dict(
                comparison["exploration_scores"]
            ),
            belief_context_id=comparison[
                "belief_context_id"
            ],
            selection_mode="counterfactual_strategy",
            strategy_scores=dict(
                comparison["strategy_scores"]
            ),
            counterfactual_rewards=dict(
                comparison["predicted_rewards"]
            ),
            state_key=(
                comparison["state_key"]
            ),
            scalar_state_key=(
                comparison[
                    "scalar_state_key"
                ]
            ),
            objective_profile_instance_id=(
                comparison[
                    "objective_profile_instance_id"
                ]
            ),
            objective_profile_signature=(
                self.resolve_objective_profile(
                    comparison[
                        "objective_profile_instance_id"
                    ]
                ).signature
            ),
        )
        self.decision_memory.append(decision)

        # Aktual: planner dipanggil lagi setelah keputusan dibuat.
        actual = self._evaluate_and_learn_route(
            decision,
            by_name[selected],
        )

        prediction_error = self.assess_prediction_error(
            context,
            comparison["counterfactuals"][selected],
            actual.outcome,
            actual_world_signature=actual.world_signature,
        )

        return StrategyExecution(
            decision=decision,
            counterfactuals=comparison["counterfactuals"],
            actual_episode=actual,
            prediction_error=prediction_error,
        )

    def counterfactual_state(self) -> Dict:
        return {
            "simulations": len(
                self.counterfactual_memory.all()
            ),
            "actual_decisions": len(
                self.decision_memory.all()
            ),
            "actual_world_episodes": len(
                self.world_decision_history
            ),
        }

    def _evaluate_and_learn_route(
        self,
        decision: DecisionRecord,
        route_action: RouteAction,
    ) -> WorldDecisionEpisode:
        actual_world_signature = (
            self._world_signature()
        )

        trajectory, audits = (
            self.plan_and_interrogate_spacetime_route(
                route_action.start,
                route_action.goal,
            )
        )

        outcome = (
            self.world_outcome_evaluator.evaluate_route(
                route_action.start,
                route_action.goal,
                trajectory,
                audits,
            )
        )

        self.record_decision_outcome(
            decision.decision_id,
            outcome.reward,
        )

        self.record_world_model_outcome(
            context=decision.context,
            action_name=decision.selected_action,
            action_instance_id=(
                decision.selected_action_instance_id
            ),
            state_key=(
                decision.state_key
                if decision.state_key is not None
                else decision.context
            ),
            reward=outcome.reward,
            success=outcome.success,
            belief_context_id=(
                decision.belief_context_id
            ),
            objective_profile_reference=(
                decision.objective_profile_instance_id
                if decision.objective_profile_instance_id
                    is not None
                else self._objective_compatibility_instance_id
            ),
        )

        world_episode = WorldDecisionEpisode(
            decision_id=decision.decision_id,
            context=decision.context,
            selected_action=(
                decision.selected_action
            ),
            trajectory=trajectory,
            audits=list(audits),
            outcome=outcome,
            world_signature=(
                actual_world_signature
            ),
            action_instance_id=(
                decision.selected_action_instance_id
            ),
            state_key=(
                decision.state_key
                if decision.state_key is not None
                else decision.context
            ),
            scalar_state_key=(
                decision.scalar_state_key
                if decision.scalar_state_key is not None
                else self._objective_scalar_state_key_from_canonical(
                    (
                        decision.state_key
                        if decision.state_key is not None
                        else decision.context
                    ),
                    objective_profile_reference=(
                        decision.objective_profile_instance_id
                        if decision.objective_profile_instance_id
                            is not None
                        else self._objective_compatibility_instance_id
                    ),
                )
            ),
            objective_profile_instance_id=(
                decision.objective_profile_instance_id
                if decision.objective_profile_instance_id
                    is not None
                else self._objective_compatibility_instance_id
            ),
        )
        self.world_decision_history.append(
            world_episode
        )
        self.maintain_memory(
            ("world_decision",)
        )
        return world_episode

    def execute_route_action(
        self,
        context: str,
        route_action: RouteAction,
        epistemic_score: float = 0.5,
        belief_context_id: Optional[str] = None,
    ) -> WorldDecisionEpisode:
        """
        Eksekusi satu action rute dan belajar langsung dari hasil planner/audit.
        Berguna untuk training episode atau action yang sudah dipilih upstream.
        """
        decision = self.choose_action(
            context,
            [route_action.name],
            epistemic_scores={
                route_action.name: epistemic_score,
            },
            belief_context_id=belief_context_id,
        )
        return self._evaluate_and_learn_route(
            decision,
            route_action,
        )

    def choose_and_execute_route(
        self,
        context: str,
        route_actions: List[RouteAction],
        epistemic_scores: Optional[Dict[str, float]] = None,
        belief_context_id: Optional[str] = None,
    ) -> WorldDecisionEpisode:
        """
        Policy memilih action TANPA melihat outcome masa depan.
        Hanya action terpilih yang diplan/diaudit/diberi reward.
        """
        if not route_actions:
            raise ValueError("route_actions tidak boleh kosong")

        by_name: Dict[str, RouteAction] = {}
        for action in route_actions:
            if action.name in by_name:
                raise ValueError(
                    f"Nama route action duplikat: {action.name}"
                )
            by_name[action.name] = action

        decision = self.choose_action(
            context,
            list(by_name),
            epistemic_scores=epistemic_scores,
            belief_context_id=belief_context_id,
        )

        selected = by_name[decision.selected_action]
        return self._evaluate_and_learn_route(
            decision,
            selected,
        )

    def world_learning_state(self) -> Dict:
        successes = sum(
            1
            for episode in self.world_decision_history
            if episode.outcome.success
        )
        rewards = [
            episode.outcome.reward
            for episode in self.world_decision_history
        ]
        return {
            "world_episodes": len(self.world_decision_history),
            "successes": successes,
            "average_reward": (
                sum(rewards) / len(rewards)
                if rewards else 0.0
            ),
        }

    def plan_and_interrogate_spacetime_route(self, start: Point, goal: Point) -> Tuple[Optional[List[SpaceTimeNode]], List[AuditReport]]:
        trajectory = self.planner.plan_spacetime_path(start, goal)
        audits = []

        if not trajectory:
            audits.append(AuditReport("Trajectory", False, "CRITICAL", "No_Path_Found", "Jalur ruang-waktu terputus/terblokir."))
            return None, audits

        # 1. Audit Tabrakan Tiap Detik
        collision = False
        for node in trajectory:
            if self.costmap.is_occupied_at(node.p, node.t):
                audits.append(AuditReport("Collision", False, "CRITICAL", "SpaceTime_Collision", f"Tabrakan di ({node.p.x},{node.p.y}) t={node.t}!"))
                collision = True
                break
        if not collision:
            audits.append(AuditReport("Collision", True, "LOW", "Collision_Free", "Jalur 100% bebas tabrakan dinamis dan statis."))

        # 2. Audit Margin Papasan Dekat
        min_d = float("inf")
        for node in trajectory:
            for obs in self.costmap.moving_obstacles:
                d = node.p.euclidean(obs.position_at(node.t))
                min_d = min(min_d, d)

        # PERBAIKAN #4: ambang audit kini konsisten dengan costmap —
        # dua tingkat: berbahaya (< COLLISION_MARGIN) dan di bawah clearance aman.
        if not self.costmap.moving_obstacles:
            audits.append(AuditReport(
                "Proximity", True, "LOW", "No_Moving_Obstacle",
                "Tidak ada rintangan bergerak; clearance dinamis tidak diperlukan."
            ))
        elif min_d < COLLISION_MARGIN:
            audits.append(AuditReport("Proximity", False, "HIGH", "Hazardous_Proximity", f"Jarak papasan terlalu mepet ({min_d:.2f} unit)."))
        elif min_d < SAFETY_CLEARANCE:
            audits.append(AuditReport("Proximity", False, "MEDIUM", "Insufficient_Clearance", f"Jarak papasan {min_d:.2f} unit di bawah clearance aman ({SAFETY_CLEARANCE} unit)."))
        else:
            audits.append(AuditReport("Proximity", True, "LOW", "Safe_Clearance", f"Margin jarak papasan aman: {min_d:.2f} unit."))

        # 3. Audit Efisiensi Menunggu (Wait Action)
        waits = sum(1 for i in range(len(trajectory) - 1) if trajectory[i].p == trajectory[i+1].p)
        transitions = max(1, len(trajectory) - 1)
        wait_ratio = waits / transitions
        if wait_ratio > 0.5:
            audits.append(AuditReport("Efficiency", False, "MEDIUM", "Excessive_Wait", f"Terlalu banyak waktu menganggur ({waits} detik)."))
        else:
            audits.append(AuditReport("Efficiency", True, "LOW", "Wait_Optimal", f"Waktu tunggu proporsional ({waits} detik dari {len(trajectory)} langkah)."))

        return trajectory, audits


# Trusted-local checkpoint compatibility.
IntegratedCognitiveAgent.__module__ = "agen_kognitif_v2_28"

__all__ = [
    "IntegratedCognitiveAgent",
    "CORE_VERSION",
    "INTEGRATION_CANDIDATE",
    "ONTOLOGY_WEIGHT",
]
