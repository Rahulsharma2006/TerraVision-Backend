
import sqlite3, json
from .config import settings

def connect():
    c = sqlite3.connect(settings.DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS scenes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT UNIQUE,
            scene_name TEXT,
            acquisition_date TEXT,
            sensor TEXT,
            platform TEXT,
            product_level TEXT,
            crs TEXT,
            width INTEGER,
            height INTEGER,
            resolution REAL,
            cloud_cover REAL,
            bounds_json TEXT,
            bands_json TEXT,
            checksum TEXT,
            quality REAL,
            provenance_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS tiles(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id INTEGER,
            tile_path TEXT UNIQUE,
            embedding_id INTEGER,
            row_off INTEGER,
            col_off INTEGER,
            width INTEGER,
            height INTEGER,
            bounds_json TEXT,
            quality REAL,
            cloud_fraction REAL,
            FOREIGN KEY(scene_id) REFERENCES scenes(id)
        );
        CREATE TABLE IF NOT EXISTS changes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            before_scene_id INTEGER,
            after_scene_id INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_id INTEGER,
            decision TEXT,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_scene_date ON scenes(acquisition_date);
        CREATE INDEX IF NOT EXISTS idx_scene_sensor ON scenes(sensor);
        CREATE INDEX IF NOT EXISTS idx_tile_scene ON tiles(scene_id);
        """)

def upsert_scene(m):
    with connect() as c:
        c.execute("""
        INSERT INTO scenes(
          source_path,scene_name,acquisition_date,sensor,platform,product_level,
          crs,width,height,resolution,cloud_cover,bounds_json,bands_json,checksum,
          quality,provenance_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source_path) DO UPDATE SET
          scene_name=excluded.scene_name,
          acquisition_date=excluded.acquisition_date,
          sensor=excluded.sensor,
          platform=excluded.platform,
          product_level=excluded.product_level,
          crs=excluded.crs,
          width=excluded.width,height=excluded.height,
          resolution=excluded.resolution,cloud_cover=excluded.cloud_cover,
          bounds_json=excluded.bounds_json,bands_json=excluded.bands_json,
          checksum=excluded.checksum,quality=excluded.quality,
          provenance_json=excluded.provenance_json
        """, (
            m["source_path"],m["scene_name"],m.get("acquisition_date"),m.get("sensor"),
            m.get("platform"),m.get("product_level"),m.get("crs"),m["width"],m["height"],
            m.get("resolution"),m.get("cloud_cover"),json.dumps(m["bounds"]),
            json.dumps(m.get("bands",[])),m.get("checksum"),m.get("quality",1.0),
            json.dumps(m.get("provenance",{}))
        ))
        return c.execute("SELECT id FROM scenes WHERE source_path=?",
                         (m["source_path"],)).fetchone()["id"]

def add_tile(scene_id,m):
    with connect() as c:
        c.execute("""
        INSERT OR REPLACE INTO tiles(
          scene_id,tile_path,embedding_id,row_off,col_off,width,height,
          bounds_json,quality,cloud_fraction
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            scene_id,m["tile_path"],m["embedding_id"],m["row_off"],m["col_off"],
            m["width"],m["height"],json.dumps(m["bounds"]),m.get("quality",1.0),
            m.get("cloud_fraction",0.0)
        ))

def list_scenes():
    with connect() as c:
        return [dict(x) for x in c.execute("SELECT * FROM scenes ORDER BY acquisition_date DESC, id DESC")]

def list_tiles():
    with connect() as c:
        rows=c.execute("""
        SELECT t.*,s.scene_name,s.acquisition_date,s.sensor,s.platform,s.cloud_cover
        FROM tiles t JOIN scenes s ON s.id=t.scene_id
        ORDER BY s.acquisition_date,t.id
        """).fetchall()
        return [dict(x) for x in rows]

def save_change(before_id,after_id,result):
    with connect() as c:
        cur=c.execute(
            "INSERT INTO changes(before_scene_id,after_scene_id,result_json) VALUES(?,?,?)",
            (before_id,after_id,json.dumps(result))
        )
        return cur.lastrowid

def list_feedback(limit=100):
    with connect() as c:
        rows=c.execute("""
        SELECT id,tile_id,decision,note,created_at
        FROM feedback
        ORDER BY id DESC
        LIMIT ?
        """, (int(limit),)).fetchall()
        return [dict(x) for x in rows]
def save_feedback(m):
    with connect() as c:
        cur=c.execute(
            "INSERT INTO feedback(tile_id,decision,note) VALUES(?,?,?)",
            (m.get("tile_id"),m["decision"],m.get("note",""))
        )
        return {"id":cur.lastrowid,"status":"saved"}

