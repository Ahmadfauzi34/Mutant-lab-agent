"""V2.39 focused sandbox — deterministic recovery policy + controlled handoff."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agen_lab as core


def obj(i,x,y,w=1.0,h=1.0,labels=()):
    return core.SpatialObject2D(
        object_id=i,
        pose=core.SpatialPose2D(x,y),
        extent=core.SpatialExtent2D(w,h),
        labels=tuple(labels),
    )


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def build_agent(root, name='recovery-lab'):
    a=core.IntegratedCognitiveAgent(
        name,10,10,
        epistemic_archive_path=str(root/(name+'.sqlite3')),
    )
    a.register_spatial_scene(
        (
            obj('BLOCK',-6,0,4,2,('movable',)),
            obj('BOX',0,0,3,5,('container',)),
        ),
        namespace='lab', scene_id='source', observed_at=0,
    )
    plan=a.plan_spatial_manipulation(
        'source',
        core.SpatialRelationGoal('BLOCK',core.SpatialRelationType.INSIDE,'BOX'),
        (
            core.SpatialManipulationOperator.rotate('BLOCK',1),
            core.SpatialManipulationOperator.place_inside('BLOCK','BOX'),
        ),
    ).best_plan
    return a,plan


def close_geometry(a,plan, *, inside=False, register=False):
    t=a.prepare_spatial_plan_execution_step(plan,1)
    a.acknowledge_spatial_execution_dispatch(
        t.ticket_id,external_receipt='adapter:'+t.ticket_id,dispatched_at=1
    )
    em=t.predicted_scene.object_map(); block=em['BLOCK']; box=em['BOX']
    if inside:
        block=obj('BLOCK',0,0,block.extent.width,block.extent.height,block.labels)
    else:
        block=obj('BLOCK',block.pose.x+0.25,block.pose.y,block.extent.width,block.extent.height,block.labels)
    actual=core.make_spatial_scene(
        (block,box),namespace='lab',belief_context_id='ctx-0',frame_id='world',
        scene_id='actual-inside' if inside else 'actual-geometry',observed_at=2,
    )
    f=a.submit_spatial_execution_observation(
        t.ticket_id,actual,register_actual_scene=register
    )
    return t,f


def main():
    with tempfile.TemporaryDirectory(prefix='v239_recovery_lab_') as td:
        root=Path(td)
        a,plan=build_agent(root)
        empirical_before={
            'q':copy.deepcopy(a.decision_policy.scoped_counts),
            'world':copy.deepcopy(a.contextual_world_model._stats),
            'joint':copy.deepcopy(a.joint_objective_model._groups),
            'evidence':len(a.all_evidence()),
            'patterns':copy.deepcopy(a.structural_patterns.patterns),
        }

        ticket,feedback=close_geometry(a,plan,register=False)
        pre=a.evaluate_spatial_recovery(plan,ticket.ticket_id)
        replan=a.replan_spatial_after_execution_deviation(
            plan,ticket.ticket_id,
            (core.SpatialManipulationOperator.place_inside('BLOCK','BOX'),),
            requested_at=2,
        )
        post=a.evaluate_spatial_recovery(
            plan,ticket.ticket_id,replan_id=replan.replan_id
        )
        handoff=a.prepare_spatial_recovery_handoff(post.recovery_id)
        handoff2=a.prepare_spatial_recovery_handoff(post.recovery_id)

        # Actual-goal-already-satisfied path.
        b,plan_b=build_agent(root,'goal-lab')
        tb,fb=close_geometry(b,plan_b,inside=True)
        rb=b.replan_spatial_after_execution_deviation(
            plan_b,tb.ticket_id,
            (core.SpatialManipulationOperator.place_inside('BLOCK','BOX'),),
        )
        db=b.evaluate_spatial_recovery(plan_b,tb.ticket_id,replan_id=rb.replan_id)

        # Exhausted path.
        c,plan_c=build_agent(root,'abort-lab')
        tc,fc=close_geometry(c,plan_c)
        rc=c.replan_spatial_after_execution_deviation(
            plan_c,tc.ticket_id,
            (core.SpatialManipulationOperator.rotate('BLOCK',1),),
            max_depth=2,
        )
        dc=c.evaluate_spatial_recovery(plan_c,tc.ticket_id,replan_id=rc.replan_id)

        # Scope mismatch -> intervention without replan.
        d,plan_d=build_agent(root,'scope-lab')
        td1=d.prepare_spatial_plan_execution_step(plan_d,1)
        d.acknowledge_spatial_execution_dispatch(td1.ticket_id,external_receipt='scope:1',dispatched_at=1)
        e=td1.predicted_scene
        scope_actual=core.make_spatial_scene(
            e.objects,namespace='other',belief_context_id=e.belief_context_id,
            frame_id=e.frame_id,scene_id='scope-actual',observed_at=2,
        )
        fd=d.submit_spatial_execution_observation(td1.ticket_id,scope_actual,register_actual_scene=False)
        dd=d.evaluate_spatial_recovery(plan_d,td1.ticket_id)

        # Match -> continue original.
        m,plan_m=build_agent(root,'match-lab')
        tm=m.prepare_spatial_plan_execution_step(plan_m,1)
        m.acknowledge_spatial_execution_dispatch(tm.ticket_id,external_receipt='match:1',dispatched_at=1)
        ee=tm.predicted_scene
        match_actual=core.make_spatial_scene(
            ee.objects,namespace=ee.namespace,belief_context_id=ee.belief_context_id,
            frame_id=ee.frame_id,scene_id='match-actual',observed_at=2,
        )
        fm=m.submit_spatial_execution_observation(tm.ticket_id,match_actual,register_actual_scene=False)
        dm=m.evaluate_spatial_recovery(plan_m,tm.ticket_id)

        empirical_after={
            'q':copy.deepcopy(a.decision_policy.scoped_counts),
            'world':copy.deepcopy(a.contextual_world_model._stats),
            'joint':copy.deepcopy(a.joint_objective_model._groups),
            'evidence':len(a.all_evidence()),
            'patterns':copy.deepcopy(a.structural_patterns.patterns),
        }

        desc=json.loads(json.dumps(post.to_descriptor()))
        portable=root/'recovery.db'
        meta=a.save_portable_state(portable)
        hash_before=sha(portable)
        restored=core.IntegratedCognitiveAgent.load_portable_state(portable)
        rd=restored.spatial_recovery_record(post.recovery_id)
        rh=restored.spatial_execution_ticket(handoff.ticket_id)
        # mutate restored runtime only
        restored.spatial_recovery=core.SpatialRecoveryStore(limit=8)
        hash_after=sha(portable)

        probe=subprocess.run(
            [sys.executable,'-c',f'''
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(Path({str(portable)!r}))
d=a.spatial_recovery_record({post.recovery_id!r})
t=a.spatial_execution_ticket({handoff.ticket_id!r})
print("RESULT="+json.dumps({{
 "version":core.CORE_VERSION,
 "action":d.action.value,
 "reason":d.reason.value,
 "handoff":t.status.value,
 "auto_replan":a.spatial_recovery_state()["automatic_replanning"],
 "auto_dispatch":a.spatial_recovery_state()["automatic_dispatch"],
 "physical":a.spatial_recovery_state()["physical_execution_performed_by_core"],
}}))
'''],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180,
        )
        line=next((x for x in probe.stdout.splitlines() if x.startswith('RESULT=')),None)
        probe_data=json.loads(line.split('=',1)[1]) if line else {}

        checks={
            'core_v2_39':core.CORE_VERSION=='2.42',
            'geometry_deviation_detected':feedback.status==core.SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION,
            'pre_replan_requests_replan':pre.action==core.SpatialRecoveryAction.REQUEST_REPLAN,
            'policy_did_not_auto_replan':a.spatial_replanning_state()['retained_records']==1,
            'replacement_found':replan.status==core.SpatialPlanningStatus.FOUND,
            'replacement_is_one_step':replan.replacement_plan.step_count==1,
            'post_replan_handoff_eligible':post.action==core.SpatialRecoveryAction.HANDOFF_REPLACEMENT,
            'handoff_not_real_world_safety_guarantee':not post.is_real_world_safety_guarantee,
            'handoff_ticket_prepared':handoff.status==core.SpatialExecutionTicketStatus.PREPARED,
            'handoff_not_dispatched':handoff.external_receipt is None,
            'handoff_idempotent':handoff is handoff2,
            'handoff_uses_replacement_plan':handoff.plan_id==replan.replacement_plan.plan_id,
            'handoff_uses_actual_unregistered_scene':handoff.source_scene_id==feedback.observed_scene.scene_id,
            'goal_already_satisfied_maps_goal_satisfied':db.action==core.SpatialRecoveryAction.GOAL_SATISFIED,
            'exhausted_maps_abort_recovery':dc.action==core.SpatialRecoveryAction.ABORT_RECOVERY,
            'scope_mismatch_requires_intervention':dd.action==core.SpatialRecoveryAction.REQUIRE_INTERVENTION,
            'match_continues_original':dm.action==core.SpatialRecoveryAction.CONTINUE_ORIGINAL,
            'decision_descriptor_json_safe':desc['schema']=='agen-spatial-recovery-decision-v1' and desc['action']=='handoff_replacement',
            'policy_version_pinned':post.policy_version=='V2.39_DETERMINISTIC_RECOVERY_V1',
            'recovery_nonexperience':not post.is_experience and not post.is_evidence and not post.was_executed,
            'recovery_does_not_train_empirical_models':empirical_before==empirical_after,
            'automatic_feedback_side_effect_false':a.spatial_recovery_state()['automatic_feedback_side_effect'] is False,
            'automatic_replanning_false':a.spatial_recovery_state()['automatic_replanning'] is False,
            'automatic_dispatch_false':a.spatial_recovery_state()['automatic_dispatch'] is False,
            'physical_execution_false':a.spatial_recovery_state()['physical_execution_performed_by_core'] is False,
            'portable_language_neutral':meta['language_neutral'] is True and meta['python_pickle'] is False,
            'portable_preserves_recovery_decision':rd.action==core.SpatialRecoveryAction.HANDOFF_REPLACEMENT,
            'portable_preserves_handoff_ticket':rh.status==core.SpatialExecutionTicketStatus.PREPARED,
            'portable_source_snapshot_immutable':hash_before==hash_after,
            'fresh_process_preserves_boundary':(
                probe.returncode==0 and probe_data.get('version')=='2.42'
                and probe_data.get('action')=='handoff_replacement'
                and probe_data.get('handoff')=='prepared'
                and probe_data.get('auto_replan') is False
                and probe_data.get('auto_dispatch') is False
                and probe_data.get('physical') is False
            ),
        }
        failed=[k for k,v in checks.items() if not v]
        print(json.dumps({
            'checks':checks,
            'pre_decision':pre.to_descriptor(),
            'post_decision':post.to_descriptor(),
            'goal_decision':db.to_descriptor(),
            'abort_decision':dc.to_descriptor(),
            'scope_decision':dd.to_descriptor(),
            'match_decision':dm.to_descriptor(),
            'fresh_process':probe_data,
        },indent=2,sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__=='__main__':
    main()
