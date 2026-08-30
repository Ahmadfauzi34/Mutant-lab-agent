"""V2.38 focused sandbox — deviation-triggered bounded spatial replanning."""
from __future__ import annotations
import copy, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import agen_lab as core

def obj(i,x,y,w=1,h=1,labels=()):
    return core.SpatialObject2D(object_id=i,pose=core.SpatialPose2D(x,y),extent=core.SpatialExtent2D(w,h),labels=tuple(labels))

def h(path):
    d=hashlib.sha256(); d.update(Path(path).read_bytes()); return d.hexdigest()

def make_plan(agent):
    agent.register_spatial_scene((obj('BLOCK',-6,0,4,2,('movable',)),obj('BOX',0,0,3,5,('container',))),namespace='lab',scene_id='source',observed_at=0)
    return agent.plan_spatial_manipulation('source',core.SpatialRelationGoal('BLOCK',core.SpatialRelationType.INSIDE,'BOX'),(core.SpatialManipulationOperator.rotate('BLOCK',1),core.SpatialManipulationOperator.place_inside('BLOCK','BOX'))).best_plan

def main():
    with tempfile.TemporaryDirectory(prefix='v239_reg_replan_lab_') as td:
        root=Path(td)
        agent=core.IntegratedCognitiveAgent('replan-lab',10,10,epistemic_archive_path=str(root/'cold.sqlite3'))
        plan=make_plan(agent)
        old_plan_sig=plan.semantic_signature
        empirical_before={
            'q':copy.deepcopy(agent.decision_policy.scoped_counts),
            'world':copy.deepcopy(agent.contextual_world_model._stats),
            'joint':copy.deepcopy(agent.joint_objective_model._groups),
            'evidence':len(agent.all_evidence()),
            'patterns':copy.deepcopy(agent.structural_patterns.patterns),
        }
        t=agent.prepare_spatial_plan_execution_step(plan,1)
        agent.acknowledge_spatial_execution_dispatch(t.ticket_id,external_receipt='robot:step1',dispatched_at=1)
        em=t.predicted_scene.object_map(); b=em['BLOCK']
        actual=core.make_spatial_scene((obj('BLOCK',b.pose.x+0.25,b.pose.y,b.extent.width,b.extent.height,b.labels),em['BOX']),namespace=t.predicted_scene.namespace,belief_context_id=t.predicted_scene.belief_context_id,frame_id=t.predicted_scene.frame_id,scene_id='actual-deviation',observed_at=2)
        f=agent.submit_spatial_execution_observation(t.ticket_id,actual)
        blocked=False
        try: agent.prepare_spatial_plan_execution_step(plan,2)
        except core.SpatialExecutionContinuationBlocked: blocked=True
        place=core.SpatialManipulationOperator.place_inside('BLOCK','BOX')
        rec=agent.replan_spatial_after_execution_deviation(plan,t.ticket_id,(place,),max_depth=4,max_nodes=64,max_solutions=3,requested_at=77)
        same=agent.replan_spatial_after_execution_deviation(plan,t.ticket_id,(place,),max_depth=4,max_nodes=64,max_solutions=3,requested_at=99)
        replacement=rec.replacement_plan
        new_ticket=agent.prepare_spatial_plan_execution_step(replacement,1)
        # Separate paths: already-satisfied / exhausted / limit.
        a2=core.IntegratedCognitiveAgent('already',8,8,epistemic_archive_path=str(root/'cold2.sqlite3')); p2=make_plan(a2); t2=a2.prepare_spatial_plan_execution_step(p2,1); a2.acknowledge_spatial_execution_dispatch(t2.ticket_id,external_receipt='robot:jump',dispatched_at=1); em2=t2.predicted_scene.object_map(); bb=em2['BLOCK']; actual_goal=core.make_spatial_scene((obj('BLOCK',0,0,bb.extent.width,bb.extent.height,bb.labels),em2['BOX']),namespace='lab',belief_context_id=t2.predicted_scene.belief_context_id,frame_id='world',scene_id='actual-goal',observed_at=2); a2.submit_spatial_execution_observation(t2.ticket_id,actual_goal); already=a2.replan_spatial_after_execution_deviation(p2,t2.ticket_id,(place,))
        a3=core.IntegratedCognitiveAgent('exhaust',8,8,epistemic_archive_path=str(root/'cold3.sqlite3')); p3=make_plan(a3); t3=a3.prepare_spatial_plan_execution_step(p3,1); a3.acknowledge_spatial_execution_dispatch(t3.ticket_id,external_receipt='robot:bad',dispatched_at=1); em3=t3.predicted_scene.object_map(); bad=core.make_spatial_scene((obj('BLOCK',-6,0,4,2,em3['BLOCK'].labels),em3['BOX']),namespace='lab',belief_context_id=t3.predicted_scene.belief_context_id,frame_id='world',scene_id='actual-unrotated',observed_at=2); a3.submit_spatial_execution_observation(t3.ticket_id,bad); exhausted=a3.replan_spatial_after_execution_deviation(p3,t3.ticket_id,(place,)); limited=a3.replan_spatial_after_execution_deviation(p3,t3.ticket_id,(core.SpatialManipulationOperator.rotate('BLOCK',1),place),max_depth=1)
        empirical_after={'q':copy.deepcopy(agent.decision_policy.scoped_counts),'world':copy.deepcopy(agent.contextual_world_model._stats),'joint':copy.deepcopy(agent.joint_objective_model._groups),'evidence':len(agent.all_evidence()),'patterns':copy.deepcopy(agent.structural_patterns.patterns)}
        portable=root/'state.db'; meta=agent.save_portable_state(portable); hash_before=h(portable); restored=core.IntegratedCognitiveAgent.load_portable_state(portable); rr=restored.spatial_replanning_record(rec.replan_id); restored.spatial_replanning=core.SpatialReplanningStore(limit=7); hash_after=h(portable)
        probe=subprocess.run([sys.executable,'-c',f'''\nimport json,sys\nfrom pathlib import Path\nsys.path.insert(0,{str(PROJECT_ROOT)!r})\nimport agen_lab as core\na=core.IntegratedCognitiveAgent.load_portable_state(Path({str(portable)!r}))\nr=a.spatial_replanning_record({rec.replan_id!r})\nt=a.prepare_spatial_plan_execution_step(r.replacement_plan,1)\nprint("RESULT="+json.dumps({{"version":core.CORE_VERSION,"status":r.status.value,"steps":r.replacement_plan.step_count,"ticket":t.status.value,"auto":a.spatial_state()["autonomous_spatial_replanning"],"physical":a.spatial_state()["physical_manipulation_execution"]}}))\n'''],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        line=next((x for x in probe.stdout.splitlines() if x.startswith('RESULT=')),None); pdata=json.loads(line.split('=',1)[1]) if line else {}
        checks={
            'core_v2_38':core.CORE_VERSION=='2.42',
            'original_plan_two_steps':plan.step_count==2,
            'deviation_classified':f.status==core.SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION,
            'old_continuation_blocked':blocked,
            'replan_found':rec.status==core.SpatialPlanningStatus.FOUND,
            'replan_uses_actual_scene':rec.actual_scene_id=='actual-deviation' and replacement.source_scene_id=='actual-deviation',
            'replan_preserves_goal':replacement.goal==plan.goal,
            'replacement_one_step':replacement.step_count==1,
            'replacement_is_place_inside':replacement.steps[0].operator.kind==core.SpatialManipulationKind.PLACE_INSIDE,
            'original_plan_immutable':plan.semantic_signature==old_plan_sig,
            'replan_nonexperience':not rec.is_experience and not rec.is_evidence and not rec.was_executed,
            'replacement_nonexecuted':not replacement.was_executed,
            'idempotent_same_request':same is rec and agent.spatial_replanning_state()['retained_records']==1,
            'requested_at_does_not_advance_clock':agent.interaction_clock==2 and rec.requested_at==77,
            'explicit_replacement_execution_handoff':new_ticket.status==core.SpatialExecutionTicketStatus.PREPARED,
            'already_satisfied_path':already.status==core.SpatialPlanningStatus.ALREADY_SATISFIED and already.replacement_plan is None,
            'exhausted_path':exhausted.status==core.SpatialPlanningStatus.EXHAUSTED,
            'limit_path':limited.status==core.SpatialPlanningStatus.LIMIT_REACHED and 'max_depth' in limited.planning_result.limit_reason,
            'replan_does_not_train_empirical_models':empirical_before==empirical_after,
            'execution_feedback_remains_closed':agent.spatial_execution_ticket(t.ticket_id).status==core.SpatialExecutionTicketStatus.CLOSED,
            'portable_language_neutral':meta['language_neutral'] is True and meta['python_pickle'] is False,
            'portable_preserves_replan':rr.replan_id==rec.replan_id and rr.replacement_plan.semantic_signature==replacement.semantic_signature,
            'portable_source_immutable':hash_before==hash_after,
            'fresh_process_loads_replan':probe.returncode==0 and pdata.get('version')=='2.42' and pdata.get('status')=='found' and pdata.get('steps')==1,
            'fresh_process_can_prepare_replacement':pdata.get('ticket')=='prepared',
            'autonomous_replanning_still_false':pdata.get('auto') is False,
            'physical_execution_still_false':pdata.get('physical') is False,
            'replan_store_bounded':agent.spatial_replanning_state()['limit']==256,
            'replan_capability_explicit':agent.spatial_state()['deviation_triggered_spatial_replanning'] is True,
            'v237_execution_model_preserved':agent.spatial_state()['spatial_execution_feedback_model']=='V2.37_EXTERNAL_DISPATCH_ACTUAL_OBSERVATION',
        }
        failed=[k for k,v in checks.items() if not v]
        print(json.dumps({'checks':checks,'replan_id':rec.replan_id,'actual_scene_signature':rec.actual_scene_signature,'replacement_plan_id':replacement.plan_id,'store':agent.spatial_replanning_state(),'fresh_process':pdata},indent=2,sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed: raise AssertionError(failed)
if __name__=='__main__': main()
