import json,re,sys
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import tiers
d=pd.read_parquet(ROOT/'review/rebuilt_items.parquet').set_index('item_id')
p=pd.read_csv(ROOT/'review/near_leaks.csv')
train=d[d.split=='train']; attacks=train[(train.label==1)&train.index.str.startswith('promptshield')]
keep=attacks.sample(1500,random_state=42).index
train=train[~train.index.isin(attacks.index)|train.index.isin(keep)]
p['actual_a']=p.a.isin(train.index);p['actual_b']=p.b.isin(train.index)
print('all near pairs',p.groupby(['a_split','b_split']).size().to_dict())
contam={s:set() for s in ['dev','test']}
for r in p.itertuples():
    if r.actual_a and r.b_split in contam:contam[r.b_split].add(r.b)
    if r.actual_b and r.a_split in contam:contam[r.a_split].add(r.a)
for s,ids in contam.items():
    print('actual train near',s,'items',len(ids));print(d.loc[sorted(ids)].groupby(['tier','label']).size().to_dict())
print('LLMail pairs',len(p[p.a.str.startswith('llmail:')&p.b.str.startswith('llmail:')]))
ll=p[p.a.str.startswith('llmail:')&p.b.str.startswith('llmail:')]
print(ll.to_string(index=False))
print('dev-test near items',len(set(p.loc[(p.a_split=='dev')&(p.b_split=='test'),'b'])|set(p.loc[(p.b_split=='dev')&(p.a_split=='test'),'a'])))
f,_=tiers.load_llmail();source={f"llmail:{r.phase}:{r.position}":r.text for r in f.itertuples()}
def shingles(text):
    w=tiers.normalise(text).split();return {' '.join(w[i:i+5]) for i in range(max(1,len(w)-4))}
print('LLMail full-text similarities')
for r in ll.itertuples():
    a,b=shingles(source[r.a]),shingles(source[r.b]);print(r.a,r.b,len(a&b)/len(a|b),'truncated',d.loc[[r.a,r.b],'truncated'].tolist())
h=json.loads((ROOT/'harness/round3_harness.json').read_text());summary=pd.read_csv(ROOT/'results/summary_full.csv',index_col=0)
changes=[]
for arm in summary.index:
    slug=re.sub('[^a-z0-9]+','-',arm.lower()).strip('-')
    frames=[pd.read_parquet(x) for x in (ROOT/'cache'/slug).glob('tier-*.parquet') if x.stem!='tier-latency']
    c=pd.concat(frames);c=c[(c.config_fp==h['arms'][arm]['config_fingerprint'])&c.error.eq('')].sort_values('ts',kind='stable').drop_duplicates('item_id',keep='last').set_index('item_id')
    z=d.join(c[['score']]);dev=z[(z.split=='dev')&(z.label==0)]
    clean=dev[~dev.index.isin(contam['dev'])]
    t=np.sort(clean.score)[len(clean)-int(np.floor(.01*len(clean)))-1]
    test=z[(z.split=='test')&~z.index.isin(contam['test'])]
    pos=test[(test.tier=='1')&(test.label==1)];neg=test[(test.tier=='3e')&(test.label==0)]
    auc=roc_auc_score(pd.concat([pos,neg]).label,pd.concat([pos,neg]).score)
    changes.append({'arm':arm,'clean_dev_benign_n':len(clean),'clean_llmail_n':len(pos),'old_t':summary.loc[arm,'t_1pct'],'new_t':t,
                    'old_recall':summary.loc[arm,'recall LLMail @t_1pct'],'new_recall':pos.score.gt(t).mean(),
                    'old_auc':summary.loc[arm,'AUROC: LLMail vs LLMail benign emails'],'new_auc':auc,
                    'old_notinject':summary.loc[arm,'FPR NotInject @t_1pct'],'new_notinject':test[test.tier=='2'].score.gt(t).mean()})
pd.DataFrame(changes).to_csv(ROOT/'review/leak_sensitivity.csv',index=False)
print(pd.DataFrame(changes).to_string(index=False))
(ROOT/'review/contaminated_ids.json').write_text(json.dumps({s:sorted(v) for s,v in contam.items()},indent=2))
