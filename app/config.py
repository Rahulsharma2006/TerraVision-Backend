
from pathlib import Path
import os

class Settings:
    ROOT=Path(__file__).resolve().parents[2]
    DATA_DIR=ROOT/"data"
    SCENES_DIR=DATA_DIR/"scenes"
    TILES_DIR=DATA_DIR/"tiles"
    TMP_DIR=DATA_DIR/"tmp"
    INDEX_DIR=ROOT/"indexes"
    REPORT_DIR=ROOT/"reports"
    DB_PATH=DATA_DIR/"sih26227.sqlite3"

    # Final deployment should point to a locally staged remote-sensing model.
    MODEL_PATH=os.getenv(
        "EMBEDDING_MODEL_PATH",
        str(ROOT/"models"/"remoteclip"/"RemoteCLIP-ViT-B-32.pt")
    )

    MODEL_NAME=os.getenv(
        "EMBEDDING_MODEL",
        "RemoteCLIP-ViT-B-32"
    )

    MODEL_BACKEND=os.getenv(
        "EMBEDDING_BACKEND",
        "remoteclip"
    )
    REMOTECLIP_ARCH=os.getenv("REMOTECLIP_ARCH","ViT-B-32")
    OFFLINE_ONLY=os.getenv("OFFLINE_ONLY","1") != "0"
    DEVICE=os.getenv("DEVICE","cpu")

    # Learned temporal change detector. Keep baseline as the safe default.
    CHANGE_BACKEND=os.getenv("CHANGE_BACKEND","baseline")
    CHANGE_MODEL_PATH=os.getenv("CHANGE_MODEL_PATH","")
    REAL_DATA_DIR=DATA_DIR/"real_sentinel2"
    TEMPORAL_DIR=DATA_DIR/"temporal_pairs"

    TILE_SIZE=int(os.getenv("TILE_SIZE","256"))
    TILE_OVERLAP=int(os.getenv("TILE_OVERLAP","32"))
    MAX_CLOUD_FRACTION=float(os.getenv("MAX_CLOUD_FRACTION","0.30"))

    def ensure_dirs(self):
        for p in [
            self.DATA_DIR,self.SCENES_DIR,self.TILES_DIR,self.TMP_DIR,
            self.INDEX_DIR,self.REPORT_DIR,self.REAL_DATA_DIR,self.TEMPORAL_DIR
        ]:
            p.mkdir(parents=True,exist_ok=True)

settings=Settings()
