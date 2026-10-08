"""Normalize approved local panels, apply frozen pairing, construct analysis tables.

No clinical eligibility or identity resolution is inferred from laboratory values.
The source custodian supplies verified anonymous groups and approved panel flags.
"""
import json,secrets,hmac,hashlib
import numpy as np
import pandas as pd
from config import ROOT,O,F,prepare_run
def stable_group(center,verified_entity,secret):
    """HMAC a custodian-verified entity; keep the secret and linkage local."""
    if center not in ('development','external') or not str(verified_entity).strip():
        raise ValueError('Verified center and entity are required')
    return center+'_'+hmac.new(secret,(center+'|'+str(verified_entity)).encode(),hashlib.sha256).hexdigest()[:20]
def main():
    prepare_run();p=O/'preprocessing_private';p.mkdir(exist_ok=True)
    if any((O/'data_v2'/f'{c}_grouped_anonymous.csv').exists() for c in ['development','external']):
        raise RuntimeError('Existing analysis tables: use a fresh repository copy')
    panels=pd.read_csv(ROOT/'local_inputs/approved_panels.csv',dtype={'group_key':str,'panel_key':str}).fillna('')
    required={'center','group_key','panel_key','panel_type','sampling_times','test_times','age_source','sex','included','identity_uncertain',*F}
    if not required<=set(panels):raise ValueError(f'Missing panel columns: {required-set(panels)}')
    if panels.panel_key.duplicated().any():raise ValueError('Panel keys must be unique')
    if not set(panels.center)<= {'development','external'}:raise ValueError('Unknown center')
    if not set(panels.panel_type)<= {'chemistry','cbc'}:raise ValueError('Unknown panel type')
    for c in ['included','identity_uncertain']:
        panels[c]=pd.to_numeric(panels[c],errors='raise')
        if not set(panels[c])<={0,1}:raise ValueError('Flags must be 0/1')
    if panels.group_key.eq('').any() or panels.groupby('group_key').center.nunique().max()>1:
        raise ValueError('Grouping keys must be complete and center-specific')
    panels['event_id']=panels.panel_key
    panels.to_csv(p/'lis_panels_PRIVATE.csv',index=False)
    selected=panels[panels.included.eq(1)]
    linkage=pd.DataFrame({'group':1,'chemistry_panel_keys':selected.panel_key.where(selected.panel_type.eq('chemistry'),''),
                          'cbc_panel_keys':selected.panel_key.where(selected.panel_type.eq('cbc'),'')})
    linkage.to_csv(p/'included_record_linkage_PRIVATE.csv',index=False)
    pd.DataFrame({'group_keys':panels.loc[panels.identity_uncertain.eq(1),'group_key'].unique()}).to_csv(p/'cohort_identity_issues_PRIVATE.csv',index=False)
    salt=p/'identity_salt_PRIVATE.txt'
    if not salt.exists():salt.write_text(secrets.token_hex(32))
    from build_episodes import main as pair_episodes
    pair_episodes()
    episodes=pd.read_csv(O/'data_v2/reconstructed_case_episodes_CANDIDATE.csv')
    coverage=pd.read_csv(p/'patient_episode_coverage.csv')
    controls=pd.read_csv(ROOT/'local_inputs/approved_controls.csv',dtype={'group_key':str})
    needed={'center','group_key','AGE','SEX',*F}
    if not needed<=set(controls):raise ValueError('Incomplete control schema')
    if controls.group_key.isna().any() or controls.group_key.duplicated().any():raise ValueError('Controls must be verified distinct people')
    if not set(controls.center)<= {'development','external'}:raise ValueError('Unknown control center')
    if set(controls.group_key)&set(episodes.group_key):raise ValueError('Case-control identity overlap')
    controls['group']=0;controls['episode_id']=controls.group_key
    columns=['center','group_key','group','episode_id','AGE','SEX',*F]
    for center in ['development','external']:
        cases=episodes[episodes.center.eq(center)];hc=controls[controls.center.eq(center)]
        eligible=set(coverage.loc[coverage.strict_earliest_eligible.eq(True),'group_key'])
        baseline=cases[cases.group_key.isin(eligible)].sort_values(['group_key','episode_date']).drop_duplicates('group_key')
        for kind,part in [('grouped',cases),('strict_baseline',baseline)]:
            pd.concat([part,hc],ignore_index=True)[columns].to_csv(O/f'data_v2/{center}_{kind}_anonymous.csv',index=False)
    from validate_inputs import validate_development,validate_external
    validate_development();validate_external()
if __name__=='__main__':main()
