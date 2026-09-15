import numpy as np
from sklearn.cluster import KMeans
from .index import VectorIndex

class EmbeddingClusterer:
    def __init__(self): self.index=VectorIndex()
    def run(self,n_clusters=8,top_k=50):
        n=min(top_k,self.index.count())
        if n < 2: return {"clusters":[],"count":n}
        if self.index.index is not None:
            X=np.vstack([self.index.index.reconstruct(i) for i in range(n)]).astype("float32")
        else:
            X=np.vstack(self.index.vectors[:n]).astype("float32")
        k=min(n_clusters,n)
        labels=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(X)
        groups={}
        for i,label in enumerate(labels):
            member=dict(self.index.meta[i])
            member["ranked_index"]=i
            groups.setdefault(int(label),[]).append(member)
        return {"count":n,"n_clusters":len(groups),"clusters":[{"cluster_id":k,"members":v} for k,v in groups.items()]}
