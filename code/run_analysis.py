"""Run the controlled-access research workflow on locally supplied analysis data."""
import argparse
from config import O
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-figures',action='store_true')
    a=p.parse_args()
    # A fresh run prevents stale checkpoint reuse after inputs have changed.
    if (O/'results_v2').exists() and any((O/'results_v2').iterdir()):
        raise RuntimeError('Research results already exist; use a fresh repository copy')
    import modeling,evaluate,tables,supplementary_tables
    modeling.main();evaluate.main();tables.main();supplementary_tables.main()
    if a.include_figures:
        import figures
        figures.main()
if __name__=='__main__':main()
