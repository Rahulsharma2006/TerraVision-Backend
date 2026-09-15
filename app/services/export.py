
import json
from pathlib import Path

def export_json(payload,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return str(path)
