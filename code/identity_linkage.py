"""Local custodian utility adapted from v2 component linkage; no name-based merging.

Input records use center, primary_key, secondary_key, alternate_key, sex.
These are confidential local tokens and are never release data.
"""
from collections import defaultdict
from preprocess import stable_group
def resolve(records,secret):
    """Return stable component groups and conflicts; unresolved rows remain blank.

    Names cannot establish identity. The custodian must also exclude unresolved
    same-name/multiple-entity cases using independent source evidence.
    """
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in records:
        if r['primary_key']:find((r['center'],r['primary_key']))
    conflicts=[]
    for fld in ['secondary_key','alternate_key']:
        buckets=defaultdict(list)
        for r in records:
            if r.get(fld) and r['primary_key']:buckets[(r['center'],r[fld])].append(r)
        for (center,key),rows in buckets.items():
            cards={r['secondary_key'] for r in rows if r.get('secondary_key')}
            sexes={r['sex'] for r in rows if r.get('sex')}
            if len(cards)>1 or len(sexes)>1:
                conflicts.append((center,fld,key));continue
            primary=sorted({r['primary_key'] for r in rows})
            for k in primary[1:]:union((center,primary[0]),(center,k))
    components=defaultdict(list)
    for k in parent:components[find(k)].append(k)
    mapping={k:stable_group(k[0],repr(sorted(v)),secret) for v in components.values() for k in v}
    return [mapping.get((r['center'],r['primary_key']),'') for r in records],conflicts
