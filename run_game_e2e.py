from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent/'sandboxes/game/cognitive_game.py'
raise SystemExit(subprocess.run([sys.executable,str(p)]).returncode)
