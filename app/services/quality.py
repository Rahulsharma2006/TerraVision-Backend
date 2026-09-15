
import numpy as np

def robust_normalize(a):
    a=np.asarray(a,dtype="float32")
    out=np.zeros_like(a,dtype="float32")
    for i in range(a.shape[0]):
        b=a[i]
        finite=np.isfinite(b)
        if not finite.any():
            continue
        lo,hi=np.percentile(b[finite],[2,98])
        out[i]=np.clip((b-lo)/max(float(hi-lo),1e-6),0,1)
    return out

def estimate_cloud_fraction(rgb):
    # Conservative heuristic only. For Sentinel-2 final deployment,
    # use the official scene/classification masks instead.
    x=np.transpose(rgb[:3],(1,2,0)).astype("float32")
    x=x/(max(float(x.max()),1.0))
    bright=x.mean(axis=2)
    low_saturation=(x.max(axis=2)-x.min(axis=2))<0.12
    return float(np.mean((bright>0.82)&low_saturation))

def quality_score(cloud_fraction, valid_fraction=1.0):
    return float(np.clip((1-cloud_fraction)*valid_fraction,0,1))
