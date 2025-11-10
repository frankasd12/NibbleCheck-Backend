# api/main.py
import os, re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: on Render, prefer the INTERNAL DB URL for the web service.
# If you ever use the EXTERNAL URL, append ?sslmode=require
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/dogfood"
)
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.30"))

STATUS_WEIGHT = {"UNSAFE": 3, "CAUTION": 2, "SAFE": 1}

app = FastAPI(
    title="NibbleCheck API",
    version="0.2.1",
    docs_url="/docs",   # root (/) is 404 by design; use /docs for UI
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def db():
    # Render POSTGRES _internal_ URLs do not need SSL; external usually needs ?sslmode=require
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    # Best effort: if pg_trgm exists, set limit; ignore errors
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT set_limit(%s);", (SIMILARITY_FLOOR,))
    except Exception:
        pass
    return conn

# ------------------------- Health & Debug -------------------------

@app.get("/health")
def health():
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/__debug/db")
def debug_db():
    """Quick check: do we have tables and the search function?"""
    out: Dict[str, Any] = {}
    with db() as conn, conn.cursor() as cur:
        cur.execute("select exists (select from pg_tables where schemaname='public' and tablename='foods');")
        out["has_foods_table"] = bool(cur.fetchone()[0])
        cur.execute("select exists (select from pg_tables where schemaname='public' and tablename='synonyms');")
        out["has_synonyms_table"] = bool(cur.fetchone()[0])
        cur.execute("""
            select exists(
              select 1
              from pg_proc p
              join pg_namespace n on n.oid = p.pronamespace
              where p.proname='search_foods_enriched' and n.nspname='public'
            );
        """)
        out["has_search_function"] = bool(cur.fetchone()[0])
        if out["has_foods_table"]:
            cur.execute("select count(*) from foods;")
            out["foods_count"] = int(cur.fetchone()[0])
        if out["has_synonyms_table"]:
            cur.execute("select count(*) from synonyms;")
            out["synonyms_count"] = int(cur.fetchone()[0])
    return out

# ------------------------- Data Endpoints -------------------------

@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    with db() as conn, conn.cursor() as cur:
        # Try the rich function first
        try:
            cur.execute(
                """
                SELECT food_id, canonical_name, group_name, default_status,
                       matched, matched_from, score
                FROM search_foods_enriched(%s, %s);
                """,
                (q, limit),
            )
            rows = cur.fetchall()
            return {
                "query": q,
                "count": len(rows),
                "results": [
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
                ],
            }
        except Exception:
            # Fallback: simple ILIKE search across foods/synonyms (no extension or function needed)
            try:
                cur.execute(
                    """
                    SELECT f.id AS food_id,
                           f.canonical_name,
                           f.group_name,
                           f.default_status,
                           COALESCE(s.name, f.canonical_name) AS matched,
                           CASE WHEN s.name IS NULL THEN 'canonical_name' ELSE 'synonyms' END AS matched_from
                    FROM foods f
                    LEFT JOIN synonyms s
                           ON s.food_id = f.id
                          AND s.name ILIKE %s
                    WHERE f.canonical_name ILIKE %s
                       OR s.name ILIKE %s
                    GROUP BY f.id, f.canonical_name, f.group_name, f.default_status, s.name
                    ORDER BY f.canonical_name
                    LIMIT %s;
                    """,
                    (f"%{q}%", f"%{q}%", f"%{q}%", limit),
                )
                rows = cur.fetchall()
                return {
                    "query": q,
                    "count": len(rows),
                    "results": [
                        {
                            "food_id": r[0],
                            "canonical_name": r[1],
                            "group_name": r[2],
                            "default_status": r[3],
                            "matched": r[4],
                            "matched_from": r[5],
                            "score": 0.0,  # no scoring in fallback
                        }
                        for r in rows
                    ],
                }
            except Exception as e2:
                raise HTTPException(500, f"Search failed: {e2}")

@app.get("/foods/{food_id}")
def food_detail(food_id: int):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, canonical_name, group_name, default_status, notes, sources
            FROM foods WHERE id=%s;
            """,
            (food_id,),
        )
        food = cur.fetchone()
        if not food:
            raise HTTPException(404, "Food not found")

        cur.execute("SELECT name FROM synonyms WHERE food_id=%s ORDER BY name;", (food_id,))
        synonyms = [r[0] for r in cur.fetchall()]

        # Rules table may not exist yet on a fresh DB; tolerate that.
        rules: List[Dict[str, Any]] = []
        try:
            cur.execute("SELECT * FROM rules WHERE food_id=%s ORDER BY id;", (food_id,))
            colnames = [d.name for d in cur.description]
            rules = [dict(zip(colnames, row)) for row in cur.fetchall()]
        except Exception:
            pass

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

# -------- CV / OCR placeholders (unchanged semantics) --------

def _pick_overall_status(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "SAFE"
    w = max(STATUS_WEIGHT.get(i.get("status"), 1) for i in items)
    for k, v in STATUS_WEIGHT.items():
        if v == w:
            return k
    return "SAFE"

@app.post("/classify/resolve")
async def classify_resolve(file: UploadFile):
    try:
        content = await file.read()
        print(f"Received file: {file.filename}, size: {len(content)} bytes")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, str(e))

_TOKEN_SPLIT_RE = re.compile(r"[,\;/\(\)\[\]\{\}\u2022•]")

def _tokenize_ingredients(s: str) -> List[str]:
    s = s.lower()
    s = _TOKEN_SPLIT_RE.sub(",", s)
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if 2 <= len(p) <= 64]
    parts = [re.sub(r"^\d+%?\s*|\s*\d+%?$", "", p).strip() for p in parts]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

@app.post("/ingredients/resolve")
def ingredients_resolve(payload: Dict[str, Any]):
    text = str(payload.get("ingredients_text", "")).strip()
    if not text:
        raise HTTPException(400, "ingredients_text is required")
    tokens = _tokenize_ingredients(text)

    hits: List[Dict[str, Any]] = []
    with db() as conn, conn.cursor() as cur:
        for t in tokens:
            try:
                cur.execute(
                    """
                    SELECT food_id, canonical_name, default_status, matched_from, score
                    FROM search_foods_enriched(%s, %s);
                    """,
                    (t, 5),
                )
                rows = cur.fetchall()
                rows = [r for r in rows if float(r[4]) >= SIMILARITY_FLOOR]
                if rows:
                    # best by status weight then score
                    best = sorted(
                        rows,
                        key=lambda r: (STATUS_WEIGHT.get(r[2], 1), float(r[4])),
                        reverse=True
                    )[0]
                    hits.append({
                        "token": t,
                        "food_id": best[0],
                        "name": best[1],
                        "status": best[2],
                        "matched_from": best[3],
                        "db_score": float(best[4]),
                    })
            except Exception:
                # Fallback: plain ILIKE
                cur.execute(
                    """
                    SELECT f.id, f.canonical_name, f.default_status
                    FROM foods f
                    LEFT JOIN synonyms s ON s.food_id=f.id
                    WHERE f.canonical_name ILIKE %s OR s.name ILIKE %s
                    LIMIT 1;
                    """,
                    (f"%{t}%", f"%{t}%"),
                )
                r = cur.fetchone()
                if r:
                    hits.append({
                        "token": t,
                        "food_id": r[0],
                        "name": r[1],
                        "status": r[2],
                        "matched_from": "fallback",
                        "db_score": 0.0,
                    })

    return {"hits": hits, "overall_status": _pick_overall_status(hits)}
