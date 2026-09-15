
from pathlib import Path
import hashlib, json
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import array_bounds
from PIL import Image
from ..config import settings
from ..db import upsert_scene,add_tile
from .retrieval import get_retriever
from .quality import robust_normalize,estimate_cloud_fraction,quality_score

def checksum(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def _date_from_tags(src):
    tags=src.tags()
    for key in ["ACQUISITION_DATE","DATE_ACQUIRED","SENSING_TIME","TIFFTAG_DATETIME"]:
        if tags.get(key):
            return str(tags[key])[:19]
    return None

def ingest_scene(path,tile_size=None,overlap=None):
    path=Path(path)
    tile_size=tile_size or settings.TILE_SIZE
    overlap=settings.TILE_OVERLAP if overlap is None else overlap
    step=max(1,tile_size-overlap)

    retriever=get_retriever()
    created=0

    with rasterio.open(path) as src:
        bands=min(src.count,3)
        scene={
            "source_path":str(path.resolve()),
            "scene_name":path.stem,
            "acquisition_date":_date_from_tags(src),
            "sensor":src.tags().get("SENSOR"),
            "platform":src.tags().get("PLATFORM"),
            "product_level":src.tags().get("PROCESSING_LEVEL"),
            "crs":str(src.crs) if src.crs else None,
            "width":src.width,"height":src.height,
            "bounds":list(src.bounds),
            "resolution":float(abs(src.transform.a)),
            "bounds":list(src.bounds),
            "bands":[src.descriptions[i] if src.descriptions[i] else f"band_{i+1}" for i in range(src.count)],
            "checksum":checksum(path),
            "quality":1.0,
            "provenance":{
                "ingest":"SIH26227 final local ingestion pipeline",
                "source_file":str(path.resolve()),
                "transform":list(src.transform)
            }
        }
        scene_id=upsert_scene(scene)

        for row in range(0,src.height,step):
            for col in range(0,src.width,step):
                h=min(tile_size,src.height-row)
                w=min(tile_size,src.width-col)
                if h<64 or w<64:
                    continue
                window=Window(col,row,w,h)
                a=src.read(indexes=list(range(1,bands+1)),window=window).astype("float32")
                valid=np.isfinite(a).all(axis=0)
                if bands==1: a=np.repeat(a,3,axis=0)
                elif bands==2: a=np.concatenate([a,a[:1]],axis=0)
                a=np.nan_to_num(a)
                n=robust_normalize(a)
                cloud=estimate_cloud_fraction((n*255).astype("uint8"))
                q=quality_score(cloud,float(valid.mean()))

                # Save a visual tile. Original geospatial scene remains the source of truth.
                rgb=(np.transpose(n[:3],(1,2,0))*255).astype("uint8")
                out=settings.TILES_DIR/f"{path.stem}_{row}_{col}.png"
                Image.fromarray(rgb).save(out)

                bounds=array_bounds(h,w,src.window_transform(window))
                vector=retriever.embedding.image(out)
                embedding_id=retriever.count()
                retriever.index.add(vector,{
                    "tile_path":str(out),
                    "scene_name":path.stem,
                    "acquisition_date":scene["acquisition_date"],
                    "sensor":scene["sensor"],
                    "platform":scene["platform"],
                    "bounds":list(bounds),
                    "quality":q,
                    "cloud_fraction":cloud
                })
                add_tile(scene_id,{
                    "tile_path":str(out),"tile_url":f"/tiles/{out.name}","embedding_id":embedding_id,
                    "row_off":row,"col_off":col,"width":w,"height":h,
                    "bounds":list(bounds),"quality":q,"cloud_fraction":cloud
                })
                created+=1
    return {"scene_id":scene_id,"tiles_created":created,
            "vectors":retriever.count(),"checksum":scene["checksum"]}
