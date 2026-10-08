"""Run the actual figure workflow, including frozen-model permutation SHAP.

Background: 40 distinct development baseline people; explain 200; five cycles;
seed 20261007. This entry point also creates manuscript figures. Requires
controlled-access analysis inputs and completed local modeling/evaluation.
"""
from figures import main
if __name__=='__main__':main()
