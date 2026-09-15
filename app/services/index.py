from pathlib import Path
import json
import numpy as np
from ..config import settings

try:
    import faiss
except Exception:
    faiss = None


class VectorIndex:
    """
    Vector retrieval service.

    Supports:
    1. Real RemoteCLIP + FAISS index
    2. Legacy NumPy/FAISS demo index as fallback
    """

    def __init__(self):
        # ---------------------------------------------------------
        # REAL REMOTECLIP INDEX
        # ---------------------------------------------------------
        self.remoteclip_dir = (
            settings.ROOT / "indexes" / "real_remoteclip"
        )

        self.remoteclip_index_path = (
            self.remoteclip_dir / "remoteclip.faiss"
        )

        self.remoteclip_vectors_path = (
            self.remoteclip_dir / "vectors.npy"
        )

        self.remoteclip_meta_path = (
            self.remoteclip_dir / "metadata.json"
        )

        # ---------------------------------------------------------
        # LEGACY INDEX
        # ---------------------------------------------------------
        self.legacy_index_path = settings.INDEX_DIR / "tiles.faiss"
        self.legacy_npy_path = settings.INDEX_DIR / "tiles_vectors.npy"
        self.legacy_meta_path = settings.INDEX_DIR / "vector_meta.json"

        self.index = None
        self.vectors = []
        self.meta = []

        self.backend = "none"

        # ---------------------------------------------------------
        # Prefer REAL RemoteCLIP index
        # ---------------------------------------------------------
        if (
            faiss
            and self.remoteclip_index_path.exists()
            and self.remoteclip_meta_path.exists()
        ):
            self.index = faiss.read_index(
                str(self.remoteclip_index_path)
            )

            self.meta = json.loads(
                self.remoteclip_meta_path.read_text(
                    encoding="utf-8"
                )
            )

            self.backend = "remoteclip"

        # ---------------------------------------------------------
        # Legacy fallback
        # ---------------------------------------------------------
        elif self.legacy_meta_path.exists():

            self.meta = json.loads(
                self.legacy_meta_path.read_text(
                    encoding="utf-8"
                )
            )

            if faiss and self.legacy_index_path.exists():
                self.index = faiss.read_index(
                    str(self.legacy_index_path)
                )

                self.backend = "legacy_faiss"

            elif self.legacy_npy_path.exists():

                arr = np.load(self.legacy_npy_path)

                self.vectors = [
                    x.astype("float32")
                    for x in arr
                ]

                self.backend = "legacy_numpy"

    # -------------------------------------------------------------
    # ADD
    # -------------------------------------------------------------
    def add(self, vector, meta):

        # Real RemoteCLIP index is treated as read-only during demo.
        if self.backend == "remoteclip":
            raise RuntimeError(
                "The staged RemoteCLIP index is read-only. "
                "Rebuild the index to add new tiles."
            )

        v = np.asarray(
            vector,
            dtype="float32"
        ).reshape(1, -1)

        if faiss:

            if self.index is None:
                self.index = faiss.IndexFlatIP(
                    v.shape[1]
                )

            self.index.add(v)

        else:
            self.vectors.append(v[0])

        self.meta.append(meta)

        self.persist()

    # -------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------
    def search(self, vector, k=10):

        n = self.count()

        if n == 0:
            return []

        v = np.asarray(
            vector,
            dtype="float32"
        ).reshape(1, -1)

        # FAISS
        if faiss and self.index is not None:

            scores, ids = self.index.search(
                v,
                min(k, n)
            )

            pairs = zip(
                scores[0],
                ids[0]
            )

        # NumPy fallback
        else:

            a = np.vstack(
                self.vectors
            )

            scores = a @ v[0]

            ids = np.argsort(
                -scores
            )[:min(k, n)]

            pairs = (
                (scores[i], i)
                for i in ids
            )

        out = []

        for score, idx in pairs:

            idx = int(idx)

            if idx >= 0 and idx < len(self.meta):

                item = dict(
                    self.meta[idx]
                )

                item["similarity"] = float(
                    score
                )

                out.append(item)

        return out

    # -------------------------------------------------------------
    # PERSIST
    # -------------------------------------------------------------
    def persist(self):

        # Never overwrite the staged real RemoteCLIP index.
        if self.backend == "remoteclip":
            return

        settings.INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            faiss
            and self.index is not None
        ):

            faiss.write_index(
                self.index,
                str(self.legacy_index_path)
            )

        elif self.vectors:

            np.save(
                self.legacy_npy_path,
                np.vstack(self.vectors)
            )

        self.legacy_meta_path.write_text(
            json.dumps(
                self.meta,
                indent=2
            ),
            encoding="utf-8"
        )

    # -------------------------------------------------------------
    # COUNT
    # -------------------------------------------------------------
    def count(self):

        if (
            faiss
            and self.index is not None
        ):
            return int(
                self.index.ntotal
            )

        return len(
            self.vectors
        )

    # -------------------------------------------------------------
    # BACKEND NAME
    # -------------------------------------------------------------
    def backend_name(self):

        return self.backend