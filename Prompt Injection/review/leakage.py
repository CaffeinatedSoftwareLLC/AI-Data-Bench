"""Reconstruct items through the source loaders without writing benchmark artifacts."""
import os, sys, json
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import tiers
llmail,funnel=tiers.llmail_items()
ps,ps_stats=tiers.promptshield_items()
rows=llmail+tiers.notinject_items()+ps+tiers.agentdojo_items()+tiers.custom_items()
for item in rows:
    item['text'],item['n_tokens']=tiers.cap(item['text'])
    item['truncated']=item['n_tokens']>tiers.MAX_TOKENS
rows+=tiers.evasion_items([r for r in rows if r['tier']=='1'],'5',tiers.COUNTS['tier5'])
rows+=tiers.evasion_items([r for r in rows if r['tier']=='3'],'5b',tiers.COUNTS['tier5b'])
for item in rows:
    if item['meta'].get('transform','clean')!='clean':
        item['text'],item['n_tokens']=tiers.cap(item['text'])
        item['truncated']=item['n_tokens']>tiers.MAX_TOKENS
items=pd.DataFrame(rows); items['text_hash']=items.text.map(tiers.sha)
manifest=pd.read_csv(ROOT/'data/items_manifest.csv.gz',dtype={'tier':str})
cols=list(manifest.columns)
a=items[cols].sort_values('item_id').reset_index(drop=True)
b=manifest.sort_values('item_id').reset_index(drop=True)
manifest_differences={c:int(a[c].ne(b[c]).sum()) for c in cols}
print('manifest differences',manifest_differences,flush=True)
print('manifest dtypes',a.dtypes.to_dict(),b.dtypes.to_dict(),flush=True)
items['norm']=items.text.map(tiers.normalise)
train_all=items[items.split=='train']
attacks=train_all[(train_all.label==1)&train_all.item_id.str.startswith('promptshield')]
keep=attacks.sample(1500,random_state=tiers.SEED).index
train=train_all[~train_all.index.isin(attacks.index)|train_all.index.isin(keep)]
def overlaps(a,b,key):
    z=b[b[key].isin(set(a[key]))]
    return {'rows':len(z),'unique':z[key].nunique(),'by_tier':z.groupby(['tier','label']).size().to_dict().__repr__(),
            'ids':z.item_id.tolist()[:30]}
report={'manifest_differences':manifest_differences,'funnel':funnel,'promptshield':ps_stats,'actual_train_rows':len(train)}
for train_name,a in [('all_train',train_all),('actual_training',train)]:
    for split in ['dev','test']:
        for key in ['text','norm']:
            report[f'{train_name}_vs_{split}_{key}']=overlaps(a,items[items.split==split],key)
for key in ['text','norm']:
    report[f'dev_vs_test_{key}']=overlaps(items[items.split=='dev'],items[items.split=='test'],key)
spread=items.groupby('group').split.nunique()
report['cross_split_groups']=spread[spread>1].to_dict()
base={r['item_id']:r for r in rows}
report['evasion_non_test_bases']=[r['item_id'] for r in rows if r['tier'] in ['5','5b'] and base[r['meta']['base_item']]['split']!='test']
report['train_hash']=__import__('hashlib').sha256(json.dumps(sorted(train.text_hash),sort_keys=True,default=str).encode()).hexdigest()[:16]
items.assign(meta=items.meta.map(json.dumps)).to_parquet(ROOT/'review/rebuilt_items.parquet',index=False)
(ROOT/'review/leakage.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
