from pathlib import Path
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage


MAX_WORKING_SIZE = 2500
REGISTRATION_SIZE = 1200

CLOUD_CLASSES = {8, 9, 10}
BAD_CLASSES = {0, 1, 3, 8, 9, 10, 11}


class TemporalAnalyzer:

    def _find_scl(self, rgb_path):
        rgb = Path(rgb_path)
        scl = rgb.parent / "SCL.jp2"
        return scl if scl.exists() else None

    def read(self, path):
        """
        Read RGB GeoTIFF efficiently.

        IMPORTANT:
        The original implementation loaded the full 10980x10980
        Sentinel-2 image into RAM. This version downsamples during
        rasterio.read(), avoiding >1 GB allocations.
        """

        with rasterio.open(Path(path)) as src:

            n = min(3, src.count)

            scale = min(
                1.0,
                MAX_WORKING_SIZE / max(src.height, src.width)
            )

            out_h = max(1, int(src.height * scale))
            out_w = max(1, int(src.width * scale))

            print(
                f"Reading {Path(path).name}: "
                f"{src.width}x{src.height} -> {out_w}x{out_h}"
            )

            arr = src.read(
                indexes=list(range(1, n + 1)),
                out_shape=(n, out_h, out_w),
                resampling=Resampling.bilinear
            ).astype(np.float32)

            if n == 1:
                arr = np.repeat(arr, 3, axis=0)

            return arr

    def robust_normalize(self, image):
        image = np.asarray(image, dtype=np.float32)

        result = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            band = image[i]

            finite = np.isfinite(band)

            if not finite.any():
                continue

            values = band[finite]

            lo, hi = np.percentile(values, [2, 98])

            scale = max(float(hi - lo), 1e-6)

            result[i] = np.clip(
                (band - lo) / scale,
                0.0,
                1.0
            )

        return result

    def resize_for_registration(self, image):
        h, w = image.shape[:2]

        scale = min(
            1.0,
            REGISTRATION_SIZE / max(h, w)
        )

        nh = max(1, int(h * scale))
        nw = max(1, int(w * scale))

        return cv2.resize(
            image,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

    def read_scl(self, scl_path, target_shape):
        """
        Read Sentinel-2 SCL at the same working resolution as RGB.
        """

        if scl_path is None:
            return None

        try:
            with rasterio.open(scl_path) as src:

                h, w = target_shape

                scl = src.read(
                    1,
                    out_shape=(h, w),
                    resampling=Resampling.nearest
                )

                return scl.astype(np.uint8)

        except Exception as exc:
            print(f"SCL read warning: {exc}")
            return None

    def register(self, before, after):
        """
        ECC affine registration on a small preview.
        """

        b = np.mean(before, axis=0).astype(np.float32)
        a = np.mean(after, axis=0).astype(np.float32)

        b_preview = self.resize_for_registration(b)
        a_preview = self.resize_for_registration(a)

        b_preview = cv2.normalize(
            b_preview,
            None,
            0,
            1,
            cv2.NORM_MINMAX
        ).astype(np.float32)

        a_preview = cv2.normalize(
            a_preview,
            None,
            0,
            1,
            cv2.NORM_MINMAX
        ).astype(np.float32)

        h, w = b_preview.shape

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6
        )

        try:

            correlation, warp_matrix = cv2.findTransformECC(
                b_preview,
                a_preview,
                warp_matrix,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                1
            )

            # Convert preview translation back to working-image scale.
            preview_scale_x = before.shape[2] / max(w, 1)
            preview_scale_y = before.shape[1] / max(h, 1)

            warp_matrix[0, 2] *= preview_scale_x
            warp_matrix[1, 2] *= preview_scale_y

            registered = cv2.warpAffine(
                after.transpose(1, 2, 0),
                warp_matrix,
                (before.shape[2], before.shape[1]),
                flags=cv2.INTER_LINEAR +
                      cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            ).transpose(2, 0, 1)

            return registered, warp_matrix, float(correlation)

        except Exception as exc:

            print(f"Registration warning: {exc}")

            return (
                after.copy(),
                np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0]
                    ],
                    dtype=np.float32
                ),
                0.0
            )

    def warp_mask(self, mask, matrix, shape):
        if mask is None:
            return None

        h, w = shape

        warped = cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST +
                  cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return warped.astype(bool)

    def adaptive_threshold(self, difference, valid_mask):
        values = difference[valid_mask]

        if values.size == 0:
            return 1.0

        median = float(np.median(values))

        mad = float(
            np.median(
                np.abs(values - median)
            )
        )

        threshold = median + 3.0 * max(
            mad,
            1e-4
        )

        return float(
            np.clip(
                threshold,
                0.05,
                0.60
            )
        )

    def build_regions(self, mask, difference):
        """
        Connected components with small-noise filtering.
        """

        cleaned = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8)
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8)
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )

        regions = []

        for label in range(1, num_labels):

            x, y, w, h, area = stats[label]

            if area < 64:
                continue

            region_mask = labels == label

            region_values = difference[region_mask]

            if region_values.size == 0:
                continue

            regions.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "relative_area": float(
                        area / max(mask.size, 1)
                    ),
                    "mean_change": float(
                        np.mean(region_values)
                    ),
                    "max_change": float(
                        np.max(region_values)
                    )
                }
            )

        regions.sort(
            key=lambda x: x["area"],
            reverse=True
        )

        return regions[:50]

    def classify_change(self, change_ratio):
        if change_ratio < 0.002:
            return "no_meaningful_change"

        if change_ratio < 0.02:
            return "localized_change"

        if change_ratio < 0.10:
            return "large_area_change"

        return "extensive_change"

    def analyze(self, before_path, after_path):

        print("\n==============================")
        print("TEMPORAL CHANGE ANALYSIS")
        print("==============================")

        print("Before:", before_path)
        print("After :", after_path)

        # --------------------------------------------------
        # 1. Efficient image loading
        # --------------------------------------------------

        before = self.read(before_path)
        after = self.read(after_path)

        # Ensure same working dimensions.
        target_h = min(
            before.shape[1],
            after.shape[1]
        )

        target_w = min(
            before.shape[2],
            after.shape[2]
        )

        if (
            before.shape[1] != target_h or
            before.shape[2] != target_w
        ):
            before = before[
                :,
                :target_h,
                :target_w
            ]

        if (
            after.shape[1] != target_h or
            after.shape[2] != target_w
        ):
            after = after[
                :,
                :target_h,
                :target_w
            ]

        # --------------------------------------------------
        # 2. Robust normalization
        # --------------------------------------------------

        before = self.robust_normalize(before)
        after = self.robust_normalize(after)

        # --------------------------------------------------
        # 3. Sentinel-2 SCL quality masks
        # --------------------------------------------------

        before_scl_path = self._find_scl(before_path)
        after_scl_path = self._find_scl(after_path)

        before_scl = self.read_scl(
            before_scl_path,
            before.shape[1:]
        )

        after_scl = self.read_scl(
            after_scl_path,
            after.shape[1:]
        )

        before_valid = np.ones(
            before.shape[1:],
            dtype=bool
        )

        after_valid = np.ones(
            after.shape[1:],
            dtype=bool
        )

        before_cloud = 0.0
        after_cloud = 0.0

        if before_scl is not None:

            before_valid = ~np.isin(
                before_scl,
                list(BAD_CLASSES)
            )

            before_cloud = float(
                np.mean(
                    np.isin(
                        before_scl,
                        list(CLOUD_CLASSES)
                    )
                )
            )

        if after_scl is not None:

            after_valid = ~np.isin(
                after_scl,
                list(BAD_CLASSES)
            )

            after_cloud = float(
                np.mean(
                    np.isin(
                        after_scl,
                        list(CLOUD_CLASSES)
                    )
                )
            )

        # --------------------------------------------------
        # 4. Registration
        # --------------------------------------------------

        registered_after, matrix, correlation = self.register(
            before,
            after
        )

        registered_after_valid = self.warp_mask(
            after_valid,
            matrix,
            before.shape[1:]
        )

        if registered_after_valid is None:
            registered_after_valid = after_valid

        valid_mask = (
            before_valid &
            registered_after_valid
        )

        valid_fraction = float(
            np.mean(valid_mask)
        )

        # --------------------------------------------------
        # 5. Temporal difference
        # --------------------------------------------------

        difference = np.mean(
            np.abs(
                before - registered_after
            ),
            axis=0
        )

        difference = difference.astype(
            np.float32
        )

        difference[~valid_mask] = 0.0

        # --------------------------------------------------
        # 6. Adaptive change threshold
        # --------------------------------------------------

        threshold = self.adaptive_threshold(
            difference,
            valid_mask
        )

        change_mask = (
            difference >= threshold
        ) & valid_mask

        # --------------------------------------------------
        # 7. Connected regions
        # --------------------------------------------------

        regions = self.build_regions(
            change_mask,
            difference
        )

        change_pixels = int(
            np.count_nonzero(change_mask)
        )

        valid_pixels = int(
            np.count_nonzero(valid_mask)
        )

        change_ratio = float(
            change_pixels /
            max(valid_pixels, 1)
        )

        change_type = self.classify_change(
            change_ratio
        )

        # --------------------------------------------------
        # 8. Confidence
        # --------------------------------------------------

        registration_score = float(
            np.clip(
                max(correlation, 0.0),
                0.0,
                1.0
            )
        )

        quality_score = float(
            np.clip(
                valid_fraction,
                0.0,
                1.0
            )
        )

        region_score = float(
            np.clip(
                min(
                    len(regions) / 10.0,
                    1.0
                ),
                0.0,
                1.0
            )
        )

        confidence = float(
            np.clip(
                0.45 * registration_score +
                0.35 * quality_score +
                0.20 * region_score,
                0.0,
                1.0
            )
        )

        # --------------------------------------------------
        # 9. Provenance
        # --------------------------------------------------

        provenance = {
            "before": str(before_path),
            "after": str(after_path),
            "before_scl": (
                str(before_scl_path)
                if before_scl_path
                else None
            ),
            "after_scl": (
                str(after_scl_path)
                if after_scl_path
                else None
            ),
            "registration": "ECC affine on downsampled imagery",
            "working_resolution": [
                int(before.shape[2]),
                int(before.shape[1])
            ],
            "normalization": "2-98 percentile robust normalization",
            "quality_mask": "Sentinel-2 SCL",
            "thresholding": "median + 3*MAD adaptive threshold",
            "morphology": "5x5 opening + closing"
        }

        print(
            f"Working size: "
            f"{before.shape[2]}x{before.shape[1]}"
        )

        print(
            f"Change ratio: {change_ratio:.4f}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            f"Registration correlation: "
            f"{correlation:.4f}"
        )

        print("==============================\n")

        return {
            "registered": True,
            "registration_transform": matrix.tolist(),
            "registration_correlation": correlation,
            "change_ratio": change_ratio,
            "change_type": change_type,
            "confidence": confidence,
            "threshold": threshold,
            "regions": regions,
            "valid_fraction": valid_fraction,
            "before_cloud_fraction": before_cloud,
            "after_cloud_fraction": after_cloud,
            "quality_mask_used": (
                before_scl is not None or
                after_scl is not None
            ),
            "working_resolution_scale": float(
                before.shape[2] / 10980.0
            ),
            "method": (
                "SCL_quality_masked_"
                "registration_plus_"
                "robust_temporal_difference"
            ),
            "confounder_note": (
                "Sentinel-2 SCL cloud, shadow, snow "
                "and invalid-pixel classes are masked "
                "when available. The detector is an "
                "explainable statistical baseline for "
                "the college-round prototype."
            ),
            "provenance": provenance
        }