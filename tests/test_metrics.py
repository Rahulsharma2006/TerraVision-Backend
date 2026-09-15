
from backend.app.services.metrics import precision_at_k,recall_at_k,average_precision,binary_metrics

def test_retrieval_metrics():
    r=["a","b","c"]; rel={"a","c"}
    assert precision_at_k(r,rel,2)==0.5
    assert recall_at_k(r,rel,2)==0.5
    assert average_precision(r,rel)>0.5

def test_binary_metrics():
    x=binary_metrics([1,1,0,0],[1,0,0,0])
    assert 0<=x["f1"]<=1
    assert 0<=x["iou"]<=1
