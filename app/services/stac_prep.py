
"""
Preparation-time STAC client.

This module is intentionally NOT used by the runtime search path.
Use it on a networked staging machine to discover public imagery, then
download/package the assets and operate offline during evaluation.
"""
import json
from urllib.request import Request,urlopen

COPERNICUS_STAC="https://stac.dataspace.copernicus.eu/v1"

def search_sentinel2(bbox,start_datetime,end_datetime,limit=20):
    body={
        "collections":["sentinel-2-l2a"],
        "bbox":bbox,
        "datetime":f"{start_datetime}/{end_datetime}",
        "limit":limit
    }
    req=Request(
        COPERNICUS_STAC+"/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"},
        method="POST"
    )
    with urlopen(req,timeout=60) as r:
        return json.loads(r.read())

if __name__=="__main__":
    print("Preparation utility only. Do not call during offline evaluation.")
