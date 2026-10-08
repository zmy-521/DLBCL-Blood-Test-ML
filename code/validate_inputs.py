"""Validate approved anonymous analysis inputs without altering their row order."""
import numpy as np
import pandas as pd
from config import O,F
def load(center,kind):
    d=pd.read_csv(O/f'data_v2/{center}_{kind}_anonymous.csv',dtype={'group_key':str})
    required={'group_key','group','center',*F}
    if not required<=set(d):raise ValueError(f'Missing input columns: {required-set(d)}')
    if d.empty or d.group_key.isna().any() or d.group_key.str.strip().eq('').any():raise ValueError('Empty grouping key')
    if set(d.center)!={center} or set(d.group)!={0,1}:raise ValueError('Center or binary label invalid')
    if d.groupby('group_key').group.nunique().max()!=1:raise ValueError('Conflicting labels within a person')
    if d[d.group.eq(0)].group_key.duplicated().any():raise ValueError('Controls must be distinct people')
    x=d[F].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if np.isinf(x).any():raise ValueError('Infinite input values')
    return d
def validate_pair(center):
    d=load(center,'grouped');b=load(center,'strict_baseline')
    if b.group_key.duplicated().any():raise ValueError('Baseline must contain one episode per person')
    if not set(b.group_key)<=set(d.group_key):raise ValueError('Baseline contains unknown people')
    # Verify that baseline values and labels are an actual subset, including NaNs.
    cols=['group_key','group',*F]
    hashes=set(pd.util.hash_pandas_object(d[cols],index=False))
    if not set(pd.util.hash_pandas_object(b[cols],index=False))<=hashes:raise ValueError('Baseline episode differs from primary data')
    return d,b
def validate_development():
    d,b=validate_pair('development')
    if d.groupby('group').group_key.nunique().min()<5:raise ValueError('Insufficient groups for outer CV')
def validate_external():
    d,b=validate_pair('external')
    keys=set(pd.read_csv(O/'data_v2/development_grouped_anonymous.csv',usecols=['group_key'],dtype=str).group_key)
    if keys&set(d.group_key):raise ValueError('Cross-center group overlap')
if __name__=='__main__':
    validate_development();validate_external();print('Analysis inputs validated')
