from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
SANDBOXES=[
    ROOT/'sandboxes/focused/replan_ranking.py',
    ROOT/'sandboxes/focused/plan_ranking.py',
    ROOT/'sandboxes/focused/reliability.py',
    ROOT/'sandboxes/focused/recovery_policy.py',
    ROOT/'sandboxes/focused/replanning.py',
    ROOT/'sandboxes/focused/execution_feedback.py',
    ROOT/'sandboxes/focused/manipulation_planning.py',
    ROOT/'sandboxes/focused/counterfactual_manipulation.py',
    ROOT/'sandboxes/focused/spatial_transform.py',
    ROOT/'sandboxes/focused/spatial_relation.py',
    ROOT/'sandboxes/focused/structural_patterns.py',
    ROOT/'sandboxes/focused/portable_state.py',
    ROOT/'sandboxes/focused/objective_experience_archive.py',
    ROOT/'sandboxes/focused/preference_aware_risk.py',
    ROOT/'sandboxes/focused/joint_objective.py',
    ROOT/'sandboxes/focused/objective_profile_shift.py',
    ROOT/'sandboxes/focused/noisy_observation.py',
    ROOT/'sandboxes/focused/state_identity.py',
    ROOT/'sandboxes/game/cognitive_game.py',
]
failed=[]
for p in SANDBOXES:
    print(f'\n=== {p.relative_to(ROOT)} ===', flush=True)
    r=subprocess.run([sys.executable,str(p)])
    if r.returncode: failed.append(str(p.relative_to(ROOT)))
if failed:
    print('\nFAILED:')
    [print('-',x) for x in failed]
    raise SystemExit(1)
print('\nALL CANONICAL FOCUSED + GAME SANDBOXES PASS')
