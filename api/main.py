# api/main.py
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: on Render, prefer the INTERNAL DB URL for the web service.
# If you ever use the EXTERNAL URL, append ?sslmode=require
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/dogfood",
)

SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.30"))  # 0.25–0.35
STATUS_WEIGHT: Dict[str, int] = {"UNSAFE": 3, "CAUTION": 2, "SAFE": 1}

# -------------------------------------------------
# DB helper
# -------------------------------------------------


@contextmanager
def db():
    conn = psycopg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# -------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------

app = FastAPI(title="NibbleCheck API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
def root():
    return {"ok": True, "service": "nibblecheck-api"}


@app.get("/__health/db")
def health_db():
    """Simple DB health-check for CI / Render."""
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {"ok": True}
    except Exception as e:  # pragma: no cover - simple health endpoint
        raise HTTPException(500, str(e))


@app.get("/__debug/db")
def debug_db():
    """Check that key tables & functions exist."""
    out: Dict[str, Any] = {}
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "select exists (select from pg_tables "
            "where schemaname='public' and tablename='foods');"
        )
        out["has_foods_table"] = bool(cur.fetchone()[0])

        cur.execute(
            "select exists (select from pg_tables "
            "where schemaname='public' and tablename='synonyms');"
        )
        out["has_synonyms_table"] = bool(cur.fetchone()[0])

        cur.execute(
            """
            select exists(
              select 1
              from pg_proc p
              join pg_namespace n on n.oid = p.pronamespace
              where p.proname='search_foods_enriched' and n.nspname='public'
            );
            """
        )
        out["has_search_function"] = bool(cur.fetchone()[0])

        if out["has_foods_table"]:
            cur.execute("select count(*) from foods;")
            out["foods_count"] = int(cur.fetchone()[0])

        if out["has_synonyms_table"]:
            cur.execute("select count(*) from synonyms;")
            out["synonyms_count"] = int(cur.fetchone()[0])

    return out


# -------------------------------------------------
# Helpers for food lookup
# -------------------------------------------------


def _pick_overall_status(items: List[Dict[str, Any]]) -> str:
    """Given a list of hits with 'status', return the worst one."""
    if not items:
        return "SAFE"
    worst = max(items, key=lambda h: STATUS_WEIGHT.get(str(h.get("status")), 1))
    return str(worst.get("status"))


_TOKEN_SPLIT_RE = re.compile(r"[;/•\n\r\t]+")


def _tokenize_ingredients(s: str) -> List[str]:
    """
    Take a long ingredients string and turn it into distinct tokens to look up.
    """
    s = s.lower()
    s = _TOKEN_SPLIT_RE.sub(",", s)
    parts = [p.strip() for p in s.split(",")]
    # Filter too-short or too-long junk
    parts = [p for p in parts if 2 <= len(p) <= 64]
    # Strip leading / trailing percentages or amounts (e.g. "20% chicken" → "chicken")
    parts = [re.sub(r"^\d+%?\s*|\s*\d+%?$", "", p).strip() for p in parts]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _resolve_tokens_against_db(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Resolve ingredient tokens *directly* against the foods + synonyms tables.

    For each token:
      1. Try to match a row in `synonyms` (synonyms.name ILIKE token).
      2. If that fails, try `foods.canonical_name` ILIKE token.
    Then attach notes + sources from foods for all matches.
    """
    hits: List[Dict[str, Any]] = []
    if not tokens:
        return hits

    with db() as conn, conn.cursor() as cur:
        for t in tokens:
            token = t.strip()
            if not token:
                continue

            # --- 1) Prefer matches in synonyms ---
            cur.execute(
                """
                SELECT f.id,
                       f.canonical_name,
                       f.default_status,
                       s.name
                FROM synonyms AS s
                JOIN foods AS f ON f.id = s.food_id
                WHERE s.name ILIKE %s
                LIMIT 1;
                """,
                (f"%{token}%",),
            )
            row = cur.fetchone()
            if row:
                hits.append(
                    {
                        "token": t,
                        "food_id": row[0],
                        "name": row[1],          # canonical_name
                        "status": row[2],        # default_status
                        "matched_from": "synonym",
                        "db_score": 1.0,
                    }
                )
                continue

            # --- 2) Fall back to canonical_name in foods ---
            cur.execute(
                """
                SELECT id, canonical_name, default_status
                FROM foods
                WHERE canonical_name ILIKE %s
                LIMIT 1;
                """,
                (f"%{token}%",),
            )
            row = cur.fetchone()
            if row:
                hits.append(
                    {
                        "token": t,
                        "food_id": row[0],
                        "name": row[1],
                        "status": row[2],
                        "matched_from": "canonical",
                        "db_score": 1.0,
                    }
                )

        # --- Attach notes + sources from foods for all matched food_ids ---
        if hits:
            food_ids = sorted({h["food_id"] for h in hits})
            cur.execute(
                """
                SELECT id,
                       notes,
                       COALESCE(sources, ARRAY[]::text[])
                FROM foods
                WHERE id = ANY(%s);
                """,
                (food_ids,),
            )
            extra: Dict[int, Dict[str, Any]] = {}
            for row in cur.fetchall():
                extra[int(row[0])] = {
                    "notes": row[1],
                    "sources": list(row[2]) if row[2] is not None else [],
                }

            for h in hits:
                e = extra.get(int(h["food_id"]))
                if e:
                    h["notes"] = e["notes"]
                    h["sources"] = e["sources"]

    return hits


# -------------------------------------------------
# Simple search / details endpoints (unchanged APIs)
# -------------------------------------------------


@app.get("/foods")
def list_foods(
    q: str = Query("", description="Optional text search over canonical names"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Lightweight search over foods table; used mainly for admin / debugging.
    """
    q = q.strip()
    with db() as conn, conn.cursor() as cur:
        if not q:
            cur.execute(
                """
                SELECT id, canonical_name, group_name, default_status
                FROM foods
                ORDER BY canonical_name
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return {
                "query": "",
                "count": len(rows),
                "results": [
                    {
                        "food_id": r[0],
                        "canonical_name": r[1],
                        "group_name": r[2],
                        "default_status": r[3],
                    }
                    for r in rows
                ],
            }

        # With query: try the trigram search helper first
        try:
            cur.execute(
                """
                SELECT food_id,
                       canonical_name,
                       group_name,
                       default_status,
                       matched,
                       matched_from,
                       score
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
        except Exception as e2:
            # Fallback: simple ILIKE search across foods/synonyms (no extension or function needed)
            cur.execute(
                """
                SELECT f.id AS food_id,
                       f.canonical_name,
                       f.group_name,
                       f.default_status,
                       COALESCE(s.name, f.canonical_name) AS matched,
                       CASE
                         WHEN s.name IS NULL THEN 'canonical'
                         ELSE 'synonym'
                       END AS matched_from,
                       1.0 AS score
                FROM foods f
                LEFT JOIN synonyms s ON s.food_id = f.id
                WHERE f.canonical_name ILIKE %s OR s.name ILIKE %s
                ORDER BY f.canonical_name
                LIMIT %s;
                """,
                (f"%{q}%", f"%{q}%", limit),
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
                ],
                "warning": f"search_foods_enriched failed: {e2}",
            }


@app.get("/foods/{food_id}")
def food_detail(food_id: int):
    """
    Fetch a single food with notes, sources, synonyms, and rules.
    """
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

        cur.execute(
            "SELECT name FROM synonyms WHERE food_id=%s ORDER BY name;", (food_id,)
        )
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


# -------------------------------------------------
# CV / OCR placeholder endpoint for image classify
# -------------------------------------------------


@app.post("/classify/resolve")
async def classify_resolve(file: UploadFile):
    """
    Placeholder for future CV/OCR pipeline.

    The mobile app already treats an empty candidate list as "no detections",
    so returning an empty list keeps things working without breaking the UI.
    """
    # For now we simply return an empty list.
    return {"candidates": []}


# -------------------------------------------------
# Ingredients text → hits
# -------------------------------------------------


@app.post("/ingredients/resolve")
def ingredients_resolve(payload: Dict[str, Any]):
    """
    Accept a big blob of label text and return per-token matches
    enriched with notes and sources from the foods table.
    """
    text = str(payload.get("ingredients_text", "")).strip()
    if not text:
        raise HTTPException(400, "ingredients_text is required")

    tokens = _tokenize_ingredients(text)
    hits = _resolve_tokens_against_db(tokens)
    return {"hits": hits, "overall_status": _pick_overall_status(hits)}


# -------------------------------------------------
# Barcode → look up ingredients → hits
# -------------------------------------------------


@app.post("/barcode/resolve")
def barcode_resolve(payload: Dict[str, Any]):
    """
    Resolve a packaged-food barcode into ingredients and then cross-reference
    each ingredient against the foods DB (foods + synonyms).
    """
    code = str(payload.get("barcode", "")).strip()
    if not code:
        raise HTTPException(400, "barcode is required")

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT display_name, ingredients_text
            FROM barcode_items
            WHERE barcode = %s;
            """,
            (code,),
        )
        row = cur.fetchone()

    if not row:
        # nicely-handled "not found" case
        return {
            "barcode": code,
            "display_name": None,
            "raw_ingredients": None,
            "hits": [],
            "overall_status": "UNKNOWN",
            "error": "barcode_not_found",
            "message": (
                "This barcode is not in our food database. "
                "It may be a non-food item or a product we haven't indexed yet."
            ),
        }

    display_name, ingredients_text = row[0], row[1] or ""
    tokens = _tokenize_ingredients(ingredients_text)
    hits = _resolve_tokens_against_db(tokens)

    return {
        "barcode": code,
        "display_name": display_name,
        "raw_ingredients": ingredients_text,
        "hits": hits,
        "overall_status": _pick_overall_status(hits),
    }
