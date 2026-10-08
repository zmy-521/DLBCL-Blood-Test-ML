"""Repository-relative paths. Research outputs remain private in ignored run/."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'run'
F = json.loads((ROOT / 'metadata/candidate_features.json').read_text(encoding='utf8'))
def prepare_run():
    for name in ['data_v2','results_v2','models_v2','tables_v2','figures_v2','supplementary_figures_v2']:
        (O/name).mkdir(parents=True,exist_ok=True)
