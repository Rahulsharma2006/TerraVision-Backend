
import numpy as np

def precision_at_k(retrieved,relevant,k):
    top=retrieved[:k]
    return sum(x in relevant for x in top)/max(1,k)

def recall_at_k(retrieved,relevant,k):
    if not relevant:return 0.0
    return sum(x in relevant for x in retrieved[:k])/len(relevant)

def average_precision(retrieved,relevant):
    if not relevant:return 0.0
    hits=0; total=0.0
    for i,x in enumerate(retrieved,1):
        if x in relevant:
            hits+=1
            total+=hits/i
    return total/len(relevant)

def binary_metrics(pred,truth):
    p=np.asarray(pred).astype(bool)
    t=np.asarray(truth).astype(bool)
    tp=np.sum(p&t); fp=np.sum(p&~t); fn=np.sum(~p&t)
    precision=tp/max(1,tp+fp)
    recall=tp/max(1,tp+fn)
    f1=2*precision*recall/max(1e-9,precision+recall)
    iou=tp/max(1,tp+fp+fn)
    return {"precision":float(precision),"recall":float(recall),
            "f1":float(f1),"iou":float(iou)}
