
from pathlib import Path
import cv2, numpy as np
import rasterio
from ..config import settings

class ChangeDetector:
    def _read(self,path):
        with rasterio.open(Path(path)) as src:
            n=min(src.count,3)
            a=src.read(indexes=list(range(1,n+1))).astype("float32")
        if a.shape[0]==1: a=np.repeat(a,3,axis=0)
        if a.shape[0]==2: a=np.concatenate([a,a[:1]],axis=0)
        return np.transpose(a[:3],(1,2,0))

    def _norm(self,a):
        out=np.zeros_like(a,dtype="float32")
        for i in range(3):
            b=a[:,:,i]
            lo,hi=np.percentile(b,[2,98])
            out[:,:,i]=np.clip((b-lo)/max(hi-lo,1e-6),0,1)
        return out

    def compare(self,before_path,after_path):
        # Learned model path is explicit and never silently falls back to a
        # fake score. If unavailable, the caller gets the deterministic baseline.
        if settings.CHANGE_BACKEND == "torchscript" and settings.CHANGE_MODEL_PATH:
            from .change_model import LearnedChangeDetector
            model=LearnedChangeDetector(settings.CHANGE_MODEL_PATH, settings.DEVICE)
            mask,p=model.predict(before_path,after_path)
            ratio=float(mask.mean())
            conf=float(np.mean(np.maximum(p,1-p)))
            if ratio < .01: label="no_meaningful_change"
            elif ratio < .08: label="localized_change"
            else: label="large_area_change"
            return {"method":"learned_torchscript_change_model","model_backend":"torchscript",
                    "model_checkpoint":str(Path(settings.CHANGE_MODEL_PATH).resolve()),
                    "change_type":label,"change_ratio":ratio,"confidence":conf,
                    "threshold":0.5,"mask_array":mask,"probability_array":p,
                    "note":"Learned model output. Validate this exact checkpoint on held-out data before reporting metrics."}
        before=self._norm(self._read(before_path))
        after=self._norm(self._read(after_path))
        if before.shape!=after.shape:
            after=cv2.resize(after,(before.shape[1],before.shape[0]))
        diff=np.mean(np.abs(after-before),axis=2)
        threshold=float(np.percentile(diff,90))
        mask=(diff>=threshold).astype("uint8")*255
        kernel=np.ones((3,3),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        ratio=float((mask>0).mean())
        if ratio<.01: label="no_meaningful_change"
        elif ratio<.08: label="localized_change"
        else: label="large_area_change"
        confidence=float(np.clip(np.std(diff)+abs(ratio-.01)*3,.01,.99))
        return {"method":"normalized_absolute_difference_baseline",
                "change_type":label,"change_ratio":ratio,
                "confidence":confidence,"threshold":threshold,
                "note":"Baseline only; validate a labelled remote-sensing change model for final SIH evaluation."}
