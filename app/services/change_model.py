"""Optional validated change-detection model adapter.

The project deliberately separates the baseline temporal detector from a learned
change model.  A learned model is only enabled when its weights and runtime are
explicitly staged.  This prevents demo scores from being presented as validated
model results.

Supported backends:
- baseline: registration + robust temporal difference (always available)
- torchscript: local TorchScript model with a documented pair-input contract

For a SIH-ready learned model, use the staging/evaluation scripts for the public
ChangeFormer V6 LEVIR-CD checkpoint, then convert/package the model for the
local runtime if desired.  The official ChangeFormer repository is research /
non-commercial licensed, so provenance must be declared in the model card.
"""
from pathlib import Path
import os
import numpy as np
from PIL import Image

class LearnedChangeDetector:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.device = device
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Change model checkpoint not found: {self.checkpoint}")
        import torch
        self.torch = torch
        self.model = torch.jit.load(str(self.checkpoint), map_location=device)
        self.model.eval()

    def _rgb(self, path, size=256):
        import rasterio
        with rasterio.open(path) as src:
            n=min(src.count,3)
            a=src.read(indexes=list(range(1,n+1))).astype("float32")
        if a.shape[0]==1: a=np.repeat(a,3,axis=0)
        if a.shape[0]==2: a=np.concatenate([a,a[:1]],axis=0)
        a=np.transpose(a[:3],(1,2,0))
        out=np.zeros_like(a,dtype=np.float32)
        for i in range(3):
            lo,hi=np.percentile(a[:,:,i],[2,98])
            out[:,:,i]=np.clip((a[:,:,i]-lo)/max(hi-lo,1e-6),0,1)
        im=Image.fromarray((out*255).astype('uint8')).resize((size,size))
        x=np.asarray(im,dtype=np.float32)/255.0
        return x

    def predict(self, before_path, after_path):
        """Run a TorchScript pair model.

        Contract: model receives a float tensor [1,6,H,W] containing normalized
        before RGB followed by normalized after RGB and returns either [1,1,H,W]
        logits/probabilities or [1,2,H,W] class logits.
        """
        b=self._rgb(before_path); a=self._rgb(after_path)
        x=np.concatenate([b,a],axis=2).transpose(2,0,1)[None]
        xt=self.torch.from_numpy(x).to(self.device)
        with self.torch.inference_mode():
            y=self.model(xt)
        if isinstance(y,(tuple,list)): y=y[0]
        y=y.detach().float().cpu().numpy()
        if y.ndim==4 and y.shape[1]==2:
            # softmax change probability
            e=np.exp(y-y.max(axis=1,keepdims=True)); p=e[:,1]/np.maximum(e.sum(axis=1),1e-6)
            p=p[0]
        elif y.ndim==4 and y.shape[1]==1:
            z=y[0,0]; p=1/(1+np.exp(-z)) if (z.min()<0 or z.max()>1) else z
        else:
            raise ValueError(f"Unsupported TorchScript output shape: {y.shape}")
        mask=(p>=0.5).astype('uint8')
        return mask,p
