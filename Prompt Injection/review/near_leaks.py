import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from datasketch import MinHash,MinHashLSH
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tiers
items=pd.read_parquet(ROOT/'review/rebuilt_items.parquet')
# Search original source items, excluding intentional clean/transformed test replicas.
base=items[~items.tier.isin(['5','5b'])].copy().reset_index(drop=True)
base['norm']=base.text.map(tiers.normalise)
grams=[{' '.join(w[i:i+5]).encode() for i in range(max(1,len(w)-4))} for w in base.norm.map(str.split)]
hashes=MinHash.bulk([list(g) for g in grams],num_perm=128,seed=1)
lsh=MinHashLSH(threshold=.6,num_perm=128)
matches=[]
for i,mh in enumerate(hashes):
    for j in lsh.query(mh):
        if base.iloc[i]['split']==base.iloc[j]['split']:continue
        sim=len(grams[i]&grams[j])/len(grams[i]|grams[j])
        if sim>=.8:
            a,b=base.iloc[i],base.iloc[j]
            matches.append({'a':a.item_id,'a_split':a['split'],'b':b.item_id,'b_split':b['split'],
                            'jaccard':sim,'estimated_jaccard':float(mh.jaccard(hashes[j]))})
    lsh.insert(i,mh)
pd.DataFrame(matches).to_csv(ROOT/'review/near_leaks.csv',index=False)
print('pairs',len(matches));print(pd.DataFrame(matches).head(25).to_string(index=False))
