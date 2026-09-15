from shapely.geometry import box

def intersects_bbox(bounds, bbox):
    if not bounds or not bbox or len(bounds) != 4 or len(bbox) != 4:
        return True
    try:
        # bounds: minx,miny,maxx,maxy; bbox: same order
        return box(*bounds).intersects(box(*bbox))
    except Exception:
        return True
