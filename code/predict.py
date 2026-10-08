"""Apply the released frozen RF13 to numeric laboratory columns."""
import argparse, json, pickle
import numpy as np
import pandas as pd
from config import ROOT
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=str,default='example_data/synthetic_example.csv')
    parser.add_argument('--output',type=str,default='outputs_example/synthetic_predictions.csv')
    args=parser.parse_args()
    features=json.loads((ROOT/'model/feature_order.json').read_text(encoding='utf8'))
    meta=json.loads((ROOT/'model/final_model_metadata.json').read_text(encoding='utf8'))
    frame=pd.read_csv(ROOT/args.input)
    missing=set(features)-set(frame.columns)
    if missing:raise ValueError(f'Missing required columns: {sorted(missing)}')
    X=frame[features].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if np.isinf(X).any():raise ValueError('Infinite values are invalid')
    with (ROOT/'model/final_rf13_model.pkl').open('rb') as handle:
        model=pickle.load(handle)  # Load only this trusted release artifact.
    if list(model.classes_)!=[0,1]:raise ValueError('Unexpected class order')
    probabilities=model.predict_proba(X)[:,1]
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row_number':np.arange(1,len(frame)+1),'probability':probabilities,
                  'above_development_cutoff':probabilities>=meta['cutoff']}).to_csv(output,index=False)
    print(f'Wrote {len(frame)} predictions to {args.output}')
if __name__=='__main__':main()
