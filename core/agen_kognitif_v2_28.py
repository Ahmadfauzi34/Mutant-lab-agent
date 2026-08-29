"""Canonical V2.42 compatibility shim after modularization M7.

The historical module name ``agen_kognitif_v2_28`` is retained for old
imports, trusted-local pickle/checkpoint lookup, and regression compatibility.
All real implementation is physically owned by ``agen_lab`` modules.
"""

CORE_VERSION = "2.42"
INTEGRATION_CANDIDATE = "V3.0-V3.2_PORT_ON_V2.24_OBSERVATION_RELIABILITY"
ONTOLOGY_WEIGHT = 0.5

from agen_lab.planning import (
    RelationType, Concept, Relation, Domain,
    KnowledgeBase, Point, SpaceTimeNode, MovingObstacle,
    SpatioTemporalCostmap, SpatioTemporalPlanner, RouteAction, RELATION_WEIGHTS,
    SAFETY_CLEARANCE, COLLISION_MARGIN,
)

from agen_lab.memory import (
    BeliefShiftDecisionMemory, EpisodeMemory, TransitionMemory, DecisionMemory,
    TrajectoryDecisionMemory, CounterfactualMemory, MetaRiskDecisionMemory, PredictionMemory,
    PredictionErrorMemory, MemoryRetentionPolicy, MemoryCompactionSummary, MemoryLifecycleManager,
    EpistemicArchivePolicy, EpistemicArchiveManager,
    ObjectiveExperienceRecord, ObjectiveExperienceConflict,
)

from agen_lab.objectives import (
    OBJECTIVE_COMPONENTS, OBJECTIVE_BENEFIT_COMPONENTS, ObjectiveOutcome, ObjectiveUtilityProfile,
    ObjectiveProfileVersionConflict, ObjectiveProfileRegistry, ObjectiveAggregation, ObjectiveUtilityAggregator,
    ContextScopedObjectiveModel, ContextScopedJointObjectiveModel, ContextScopedSuccessConstraintModel,
)

from agen_lab.identity import (
    StateCanonicalDefinition, ResolvedStateIdentity, StateIdentityConflict, StateIdentityRegistry,
    ActionDefinition, ActionVersionConflict, ActionRegistry, ResolvedActionIdentity,
    ObjectiveUtilityProfile, ObjectiveProfileVersionConflict, ObjectiveProfileRegistry,
)

from agen_lab.epistemic import (
    SourceProfile, Evidence, BeliefContext, BeliefContextManager,
    BeliefShiftCandidate, BeliefShiftDecisionRecord, ContextualBeliefRevisionPolicy, SourceLineage,
    EvidenceAggregator, GroundedFact, GroundingStore, Rule,
    RuleVersionConflict, RuleValidator, Justification, RelevanceSlicer,
    ProvenanceGraph, TruthEvaluator, EpistemicVerdict, AuditReport,
    AdmissionStatus, Episode, KnowledgeAdmissionPolicy, DependencyGraph,
    TruthMaintenanceSystem, EvidenceAuditMode, IndexedEvidenceAggregate, ExactEvidenceQueryEngine,
    BeliefShiftDecisionMemory, EpisodeMemory,
)

from agen_lab.decision import (
    DecisionRecord, TransitionRecord, TrajectoryDecisionRecord, DecisionPolicy,
    UncertaintyDecisionMode, UncertaintyRiskProfile, UncertaintyDecisionResult, MetaRiskSignals,
    MetaRiskDecision, AdaptiveRiskModePolicy, UncertaintyAwareDecisionPolicy, RiskMode,
    ActionRiskEstimate, SafetyGateAssessment, ChanceConstrainedSafetyGate, MetaRiskPolicy,
    TrajectoryRiskEstimate, TrajectorySafetyAssessment, TrajectoryChanceConstrainedSafetyGate, TrajectoryRiskPolicy,
    PreferenceAwareUtilityEstimate, PreferenceAwareRiskPolicy,
    PreferenceAwareTrajectoryEstimate, PreferenceAwareTrajectoryRiskPolicy,
    TransitionMemory, DecisionMemory, TrajectoryDecisionMemory, MetaRiskDecisionMemory,
    RouteAction,
)

from agen_lab.world_model import (
    WorldOutcome, PredictionErrorRecord, PredictionUncertainty, PredictionUncertaintyEstimator,
    OutcomePrediction, ContextScopedWorldModel, WorldModelReliability, PredictionErrorEvaluator,
    CounterfactualStrategyPolicy, EnsembleMemberState, EnsemblePrediction, OnlineBootstrapEnsemble,
    ConformalInterval, OnlineConformalCalibrator, WorldDecisionEpisode, WorldOutcomeEvaluator,
    CounterfactualEstimate, StrategyExecution, CounterfactualMemory, PredictionMemory,
    PredictionErrorMemory,
)

from agen_lab.persistence import (
    PERSISTENCE_SCHEMA_VERSION, PERSISTENCE_MAGIC, AgentPersistenceError, AgentPersistenceManager,
)

from agen_lab.portable_state import (
    PORTABLE_STATE_MAGIC, PORTABLE_STATE_SCHEMA_VERSION, PORTABLE_GRAPH_SCHEMA,
    PortableStateError, PortableStateSchemaError, PortableStateTypeError,
    PortableObjectGraphCodec, PortableCognitiveStateManager,
)

from agen_lab.patterns import (
    PatternError, PatternSourceConflict, PatternPredictionError,
    PatternKind, PatternRelationType, StructuralPatternCandidate,
    StructuralPatternDefinition, StructuralPatternHypothesis,
    StructuralPatternInstance, StructuralPatternPrediction,
    StructuralPatternPredictionAssessment, PatternRelation,
    StructuralPatternEngine, StructuralPatternStore,
    canonicalize_symbol_sequence, MAX_PATTERN_SEQUENCE_LENGTH,
)

from agen_lab.spatial import (
    MAX_SPATIAL_OBJECTS_PER_SCENE, DEFAULT_SPATIAL_SCENE_LIMIT,
    MAX_SPATIAL_RELATIONS, SpatialError, SpatialSceneConflict,
    SpatialRelationConflict, SpatialRelationType, SpatialRelationSource,
    SpatialPose2D, SpatialExtent2D, SpatialBounds2D, SpatialObject2D,
    SpatialScene2D, SpatialRelation, SpatialRelationAlgebra,
    SpatialGeometry2D, SpatialSceneCanonicalizer, SpatialSceneStore,
    make_spatial_scene,
)

from agen_lab.spatial_transform import (
    DEFAULT_TRANSFORM_MATCH_TOLERANCE, MAX_TRANSFORM_MATCH_OBJECTS,
    SpatialTransformError, SpatialTransformFrameError,
    SpatialTransformMatchError, SpatialLinearTransformKind,
    SpatialTransform2D, SpatialTransformMatch, SpatialTransformInference,
    SpatialTransformationMatcher, spatial_transform_token,
)

from agen_lab.spatial_manipulation import (
    MAX_MANIPULATION_SCENE_OBJECTS, SpatialManipulationError,
    SpatialManipulationKind, SpatialManipulationCheckKind,
    SpatialManipulationOperator, SpatialManipulationCheck,
    SpatialManipulationCollision, CounterfactualSpatialManipulation,
    SpatialManipulationSimulator, spatial_manipulation_token,
)

from agen_lab.spatial_planning import (
    DEFAULT_SPATIAL_PLAN_MAX_DEPTH, DEFAULT_SPATIAL_PLAN_MAX_NODES,
    DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS, MAX_SPATIAL_PLAN_OPERATOR_CATALOG,
    SpatialPlanningError, SpatialPlanningStatus, SpatialRelationGoal,
    SpatialManipulationPlanStep, SpatialManipulationPlan,
    SpatialManipulationPlanningResult, BoundedSpatialManipulationPlanner,
    spatial_plan_token,
)

from agen_lab.spatial_execution import (
    DEFAULT_SPATIAL_EXECUTION_TICKET_LIMIT,
    DEFAULT_SPATIAL_EXECUTION_MATCH_TOLERANCE,
    SpatialExecutionError, SpatialExecutionConflict,
    SpatialExecutionStaleSource, SpatialExecutionContinuationBlocked,
    SpatialExecutionTicketStatus, SpatialExecutionFeedbackStatus,
    SpatialExecutionTicket, SpatialExecutionFeedback,
    SpatialExecutionComparator, SpatialExecutionStore,
)

from agen_lab.spatial_replanning import (
    DEFAULT_SPATIAL_REPLAN_RECORD_LIMIT,
    SpatialReplanningError, SpatialReplanningConflict,
    SpatialReplanningTriggerStatus, SpatialReplanningRecord,
    SpatialReplanningStore, DeviationTriggeredSpatialReplanner,
)

from agen_lab.spatial_recovery import (
    DEFAULT_SPATIAL_RECOVERY_RECORD_LIMIT,
    DEFAULT_SPATIAL_RECOVERY_MAX_HANDOFF_STEPS,
    SPATIAL_RECOVERY_POLICY_VERSION, SpatialRecoveryError,
    SpatialRecoveryConflict, SpatialRecoveryAction, SpatialRecoveryReason,
    SpatialRecoveryDecisionRecord, DeterministicSpatialRecoveryPolicy,
    SpatialRecoveryStore,
)

from agen_lab.spatial_reliability import (
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

from agen_lab.spatial_plan_ranking import (
    DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
    DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
    SpatialPlanRankingError, SpatialPlanRankingConflict,
    SpatialPlanReliabilityRankingStatus, SpatialPlanReliabilityCandidate,
    SpatialReliabilityRankedPlanningResult, SpatialPlanReliabilityRanker,
)

from agen_lab.spatial_replan_ranking import (
    DEFAULT_SPATIAL_REPLAN_RANKING_MIN_SAMPLES,
    DEFAULT_SPATIAL_REPLAN_RANKING_WILSON_Z,
    SpatialReplanRankingError, SpatialReplanRankingConflict,
    SpatialReliabilityRankedReplanView, SpatialReplanReliabilityRanker,
)

from agen_lab.agent import IntegratedCognitiveAgent
