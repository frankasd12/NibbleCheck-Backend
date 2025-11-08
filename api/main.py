# api/main.py
import os, re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dogfood")
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.30"))  # tune 0.25–0.35

STATUS_WEIGHT = {"UNSAFE": 3, "CAUTION": 2, "SAFE": 1}  # worst-case wins

app = FastAPI(title="NibbleCheck API", version="0.2.0")

# CORS for your dev apps; tighten for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def db():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    # set pg_trgm similarity floor for this session
    with conn.cursor() as cur:
        cur.execute("SELECT set_limit(%s);", (SIMILARITY_FLOOR,))
    return conn

@app.get("/health")
def health():
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    sql = """
      SELECT food_id, canonical_name, group_name, default_status, matched, matched_from, score
      FROM search_foods_enriched(%s, %s);
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, (q, limit))
        rows = cur.fetchall()
    results = [
        {
            "food_id": r[0],
            "canonical_name": r[1],
            "group_name": r[2],
            "default_status": r[3],
            "matched": r[4],
            "matched_from": r[5],
            "score": float(r[6]),
        }
        for r in rows
        if float(r[6]) >= SIMILARITY_FLOOR
    ]
    return {"query": q, "count": len(results), "results": results}

@app.get("/foods/{food_id}")
def food_detail(food_id: int):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, canonical_name, group_name, default_status, notes, sources
            FROM foods WHERE id=%s;
        """, (food_id,))
        food = cur.fetchone()
        if not food:
            raise HTTPException(status_code=404, detail="Food not found")

        cur.execute("SELECT name FROM synonyms WHERE food_id=%s ORDER BY name;", (food_id,))
        synonyms = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT * FROM rules WHERE food_id=%s ORDER BY id;", (food_id,))
        colnames = [desc.name for desc in cur.description]
        rules = [dict(zip(colnames, row)) for row in cur.fetchall()]

    return {
        "id": food[0],
        "canonical_name": food[1],
        "group_name": food[2],
        "default_status": food[3],
        "notes": food[4],
        "sources": food[5],
        "synonyms": synonyms,
        "rules": rules,
    }

# ---- New: CV and OCR endpoints ---------------------------------------------

def _pick_overall_status(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "SAFE"
    w = max(STATUS_WEIGHT.get(i["status"], 1) for i in items)
    for k, v in STATUS_WEIGHT.items():
        if v == w:
            return k
    return "SAFE"

@app.post("/classify/resolve")
def classify_resolve(payload: Dict[str, Any]):
    """
    Input:
    {
      "labels": [{"name": "grape", "score": 0.81}, {"name":"grapefruit","score":0.41}]
    }
    """
    labels = payload.get("labels", [])
    if not isinstance(labels, list) or not labels:
        raise HTTPException(400, "labels must be a non-empty list")

    results: List[Dict[str, Any]] = []
    with db() as conn, conn.cursor() as cur:
        for item in labels:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            cur.execute("""
              SELECT food_id, canonical_name, default_status, matched_from, score
              FROM search_foods_enriched(%s, %s);
            """, (name, 5))
            rows = [
                {
                    "food_id": r[0],
                    "name": r[1],
                    "status": r[2],
                    "matched_from": r[3],
                    "db_score": float(r[4]),
                }
                for r in cur.fetchall()
                if float(r[4]) >= SIMILARITY_FLOOR
            ]
            if rows:
                best = sorted(rows, key=lambda x: (STATUS_WEIGHT.get(x["status"], 1), x["db_score"]), reverse=True)[0]
                best["model_label"] = name
                best["model_score"] = float(item.get("score", 0))
                results.append(best)

    return {
        "overall_status": _pick_overall_status(results),
        "candidates": results
    }

_TOKEN_SPLIT_RE = re.compile(r"[,\;/\(\)\[\]\{\}\u2022•]")

def _tokenize_ingredients(s: str) -> List[str]:
    s = s.lower()
    s = _TOKEN_SPLIT_RE.sub(",", s)
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if 2 <= len(p) <= 64]
    # trim trailing % and numbers, very light normalization
    parts = [re.sub(r"^\d+%?\s*|\s*\d+%?$", "", p).strip() for p in parts]
    # drop empties and dupes
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

@app.post("/ingredients/resolve")
def ingredients_resolve(payload: Dict[str, Any]):
    """
    Input:
    { "ingredients_text": "wheat flour, raisins, cinnamon, sugar" }
    """
    text = str(payload.get("ingredients_text", "")).strip()
    if not text:
        raise HTTPException(400, "ingredients_text is required")
    tokens = _tokenize_ingredients(text)

    hits: List[Dict[str, Any]] = []
    with db() as conn, conn.cursor() as cur:
        for t in tokens:
            cur.execute("""
              SELECT food_id, canonical_name, default_status, matched_from, score
              FROM search_foods_enriched(%s, %s);
            """, (t, 5))
            rows = [
                {
                    "token": t,
                    "food_id": r[0],
                    "name": r[1],
                    "status": r[2],
                    "matched_from": r[3],
                    "db_score": float(r[4]),
                }
                for r in cur.fetchall()
                if float(r[4]) >= SIMILARITY_FLOOR
            ]
            if rows:
                best = sorted(rows, key=lambda x: (STATUS_WEIGHT.get(x["status"], 1), x["db_score"]), reverse=True)[0]
                hits.append(best)

    return {
        "hits": hits,
        "overall_status": _pick_overall_status(hits)
    }
