"""Adapted v2 pairing and chronology logic; normalized local inputs only."""
def main():
    """Use only source panels traced to originally included records. No arbitrary time pairing."""
    from pathlib import Path
    import sys,json,collections,hmac,hashlib
    import pandas as pd,numpy as np
    from config import O,F,prepare_run
    prepare_run();P=O/'preprocessing_private';A=P;D=O/'data_v2'
    m=pd.read_csv(P/'included_record_linkage_PRIVATE.csv').fillna('');pan=pd.read_csv(P/'lis_panels_PRIVATE.csv',dtype={'event_id':str,'age_source':str,'sex':str,'group_key':str,'sampling_times':str,'test_times':str}).fillna('');pan=pan.set_index('panel_key',drop=False);CHEM=F[:16];CBC=F[16:]
    issues=pd.read_csv(P/'cohort_identity_issues_PRIVATE.csv');uncertain=set('|'.join(issues.group_keys).split('|'));salt=(P/'identity_salt_PRIVATE.txt').read_text().strip().encode()
    def tok(s):return hmac.new(salt,s.encode(),hashlib.sha256).hexdigest()[:20]
    def timevals(s):return [pd.Timestamp(v) for v in str(s).split('|') if v and v!='nan']
    def datekey(r):
     ts=timevals(r.sampling_times)
     if len({x.date() for x in ts})==1:return str(ts[0].date()),'sampling'
     ts=timevals(r.test_times)
     if len({x.date() for x in ts})==1:return str(ts[0].date()),'test_proxy'
     return '', 'unavailable'
    allowed=set()
    for r in m[m.group.eq(1)].itertuples():
     allowed.update(filter(None,r.chemistry_panel_keys.split('|')));allowed.update(filter(None,r.cbc_panel_keys.split('|')))
    selected=pan.loc[sorted(allowed)].copy();selected['episode_date']=[datekey(r)[0] for r in selected.itertuples()];selected['date_basis']=[datekey(r)[1] for r in selected.itertuples()]
    selected['uncertain_identity']=selected.group_key.isin(uncertain)
    selected.to_csv(P/'included_source_panels_PRIVATE.csv',index=False,encoding='utf-8-sig')
    episodes=[];unpaired=[]
    for (center,pid,date),g in selected[~selected.uncertain_identity&selected.episode_date.ne('')].groupby(['center','group_key','episode_date']):
     ch=g[g.panel_type.eq('chemistry')];cb=g[g.panel_type.eq('cbc')]
     if len(ch)!=1 or len(cb)!=1:
      unpaired.append(dict(center=center,group_key=pid,episode_date=date,chemistry_panels=len(ch),cbc_panels=len(cb),reason='missing panel type' if min(len(ch),len(cb))==0 else 'multiple same-day panels; no arbitrary pairing'));continue
     c=ch.iloc[0];b=cb.iloc[0];cs=timevals(c.sampling_times);bs=timevals(b.sampling_times)
     strict=c.date_basis=='sampling' and b.date_basis=='sampling'
     delta=min((abs((x-y).total_seconds()/3600) for x in cs for y in bs),default=np.nan)
     # Same calendar day is the explicit provisional laboratory episode definition.
     row=dict(center=center,group_key=pid,group=1,episode_id=tok(c.panel_key+'|'+b.panel_key),episode_date=date,strict_sampling_time=strict,sampling_gap_hours=delta,chemistry_panel_key=c.panel_key,cbc_panel_key=b.panel_key)
     for f in CHEM:row[f]=pd.to_numeric(c[f],errors='coerce')
     for f in CBC:row[f]=pd.to_numeric(b[f],errors='coerce')
     # Age/sex only if sources agree; age may differ by birthday, so retain source-specific for review.
     agec=set(c.age_source.split('|'));ageb=set(b.age_source.split('|'));age=next(iter(agec)) if len(agec)==1 and agec==ageb else ''
     row['AGE']=pd.to_numeric(str(age),errors='coerce');row['SEX']=c.sex if c.sex==b.sex else ''
     episodes.append(row)
    ep=pd.DataFrame(episodes);ep.to_csv(D/'reconstructed_case_episodes_CANDIDATE.csv',index=False,encoding='utf-8-sig');pd.DataFrame(unpaired).to_csv(A/'unpaired_episode_dates.csv',index=False,encoding='utf-8-sig')
    coverage=[];patientrows=[]
    for center,g in selected.groupby('center'):
     for pid,z in g.groupby('group_key'):
      e=ep[ep.group_key.eq(pid)];all_dates=set(z.episode_date)-{''};all_sampling=bool(z.date_basis.eq('sampling').all());stricts=e[e.strict_sampling_time]
      # Earliest qualified panel within observed originally included data, not first-ever clinical sample.
      first=e.sort_values('episode_date').head(1)
      strict_first=bool(len(first) and first.strict_sampling_time.iloc[0] and all_sampling)
      patientrows.append(dict(center=center,group_key=pid,included_chemistry_panels=int(z.panel_type.eq('chemistry').sum()),included_cbc_panels=int(z.panel_type.eq('cbc').sum()),paired_episodes=len(e),strict_sampling_episodes=len(stricts),all_included_source_panels_have_sampling_time=all_sampling,strict_earliest_eligible= strict_first,identity_uncertain=pid in uncertain,reason=('identity needs source confirmation' if pid in uncertain else 'no confirmed paired episode' if e.empty else 'not all included sampling times available' if not all_sampling else 'eligible')))
     n=pd.DataFrame(patientrows);n=n[n.center.eq(center)]
     coverage.append(dict(center=center,source_identity_components=n.group_key.nunique(),uncertain_identity_components=int(n.identity_uncertain.sum()),patients_with_any_paired_episode=int(n.paired_episodes.gt(0).sum()),patients_with_strict_sampling_episode=int(n.strict_sampling_episodes.gt(0).sum()),patients_with_strict_earliest=int(n.strict_earliest_eligible.sum()),paired_episodes=int(n.paired_episodes.sum())))
    pd.DataFrame(patientrows).to_csv(A/'patient_episode_coverage.csv',index=False,encoding='utf-8-sig');(A/'episode_coverage_summary.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf8')
    print('COVERAGE',coverage);print('PAIRED',ep.groupby(['center','strict_sampling_time']).size().to_dict())

if __name__=='__main__':main()
