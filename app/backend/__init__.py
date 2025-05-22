import sys
from pathlib import Path

project_root_path = Path(__file__).resolve().parent.parent.parent
if str(project_root_path) not in sys.path:
    sys.path.append(str(project_root_path))

print(f"PYTHONPATH updated by app/backend/__init__.py to include: {project_root_path}")