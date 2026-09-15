from datetime import datetime

def pair_observations(observations, max_gap_days=None):
    obs=sorted(observations,key=lambda x:x['date'])
    pairs=[]
    for a,b in zip(obs,obs[1:]):
        if max_gap_days is not None:
            da=datetime.fromisoformat(a['date'][:10]); db=datetime.fromisoformat(b['date'][:10])
            if (db-da).days > max_gap_days: continue
        pairs.append({'before':a,'after':b,'gap_days':(datetime.fromisoformat(b['date'][:10])-datetime.fromisoformat(a['date'][:10])).days})
    return pairs
