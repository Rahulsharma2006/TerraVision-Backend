from pathlib import Path
import numpy as np
import cv2
import rasterio
from PIL import Image
from ..config import settings

def change_heatmap(before_path, after_path):
    with rasterio.open(before_path) as a, rasterio.open(after_path) as b:
        h=min(a.height,b.height); w=min(a.width,b.width)
        x=a.read(1,out_shape=(h,w)).astype('float32')
        y=b.read(1,out_shape=(h,w)).astype('float32')
    x=np.nan_to_num(x); y=np.nan_to_num(y)
    def norm(z):
        lo,hi=np.percentile(z,[2,98]); return np.clip((z-lo)/(hi-lo+1e-6),0,1)
    d=np.abs(norm(x)-norm(y))
    d=cv2.GaussianBlur(d,(0,0),1.2)
    u=np.uint8(np.clip(d*255,0,255))
    heat=cv2.applyColorMap(u,cv2.COLORMAP_TURBO)
    out=settings.TMP_DIR/'latest_change_heatmap.png'
    Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)).save(out)
    return out
