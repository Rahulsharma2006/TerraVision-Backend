
from pathlib import Path
import math
from .embedding import EmbeddingService
from .index import VectorIndex
from ..config import settings
from .aoi import intersects_bbox

_instance=None

class Retriever:
    def __init__(self):
        model=settings.MODEL_PATH or settings.MODEL_NAME
        self.embedding=EmbeddingService(model,settings.DEVICE,settings.MODEL_BACKEND)
        self.model_name=model
        self.device=self.embedding.device
        self.index=VectorIndex()

    def count(self):
        return self.index.count()

    @staticmethod
    def _date_ok(x,start_date,end_date):
        d=x.get("acquisition_date")
        if not d: return True
        if start_date and d<start_date: return False
        if end_date and d>end_date: return False
        return True

    def _rerank(self,candidates):
        # Small deterministic quality-aware reranker.
        # Keeps semantic similarity dominant while penalizing poor-quality tiles.
        out=[]
        for x in candidates:
            sim=float(x.get("similarity",0))
            q=float(x.get("quality",1))
            cloud=float(x.get("cloud_fraction",0))
            x["semantic_score"]=sim
            x["quality_score"]=q
            if x.get("tile_path"):
                x["tile_url"]="/tiles/" + Path(x["tile_path"]).name
            x["final_score"]=0.85*sim+0.15*(q*(1-cloud))
            out.append(x)
        return sorted(out,key=lambda z:z["final_score"],reverse=True)

    def search_text(self,query,k,sensor=None,start_date=None,end_date=None,
                    min_quality=0.0,aoi_bbox=None):
        candidates=self.index.search(self.embedding.text(query),max(k*10,100))
        filtered=[
            x for x in candidates
            if (not sensor or x.get("sensor")==sensor)
            and self._date_ok(x,start_date,end_date)
            and float(x.get("quality",1))>=min_quality
            and intersects_bbox(x.get("bounds"), aoi_bbox)
        ]
        return self._rerank(filtered)[:k]

    def search_image(self,path,k,aoi_bbox=None):
        candidates=self.index.search(self.embedding.image(Path(path)),max(k*10,100))
        candidates=[x for x in candidates if intersects_bbox(x.get("bounds"), aoi_bbox)]
        return self._rerank(candidates)[:k]

def get_retriever():
    global _instance
    if _instance is None:
        _instance=Retriever()
    return _instance
