# api/main.py
import os
import re
import io
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv
import requests
from PIL import Image

import vision


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
    Resolve ingredient tokens with fuzzy matching support.
    """
    hits: List[Dict[str, Any]] = []
    if not tokens:
        return hits

    with db() as conn, conn.cursor() as cur:
        for t in tokens:
            token = t.strip()
            if not token:
                continue

            # --- 1) Prefer exact/prefix matches in synonyms ---
            cur.execute(
                """
                SELECT f.id,
                       f.canonical_name,
                       f.default_status,
                       s.name,
                       similarity(s.name, %s) AS score
                FROM synonyms AS s
                JOIN foods AS f ON f.id = s.food_id
                WHERE s.name %% %s
                ORDER BY score DESC
                LIMIT 1;
                """,
                (token, token),
            )
            row = cur.fetchone()
            if row and row[4] >= SIMILARITY_FLOOR:  # Use similarity threshold
                hits.append(
                    {
                        "token": t,
                        "food_id": row[0],
                        "name": row[1],
                        "status": row[2],
                        "matched_from": "synonym",
                        "db_score": row[4],
                    }
                )
                continue

            # --- 2) Fall back to fuzzy canonical_name ---
            cur.execute(
                """
                SELECT id,
                       canonical_name,
                       default_status,
                       similarity(canonical_name, %s) AS score
                FROM foods
                WHERE canonical_name %% %s
                ORDER BY score DESC
                LIMIT 1;
                """,
                (token, token),
            )
            row = cur.fetchone()
            if row and row[3] >= SIMILARITY_FLOOR:
                hits.append(
                    {
                        "token": t,
                        "food_id": row[0],
                        "name": row[1],
                        "status": row[2],
                        "matched_from": "food",
                        "db_score": row[3],
                    }
                )

        # --- Attach notes + sources ---
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
# Simple search / details endpoints
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
    Image → YOLO detector → DB resolver.

    1) Accept an uploaded image file.
    2) Run YOLO to detect food objects.
    3) Map detected labels into canonical foods via _resolve_tokens_against_db.
    4) Return candidates + overall_status (and log to inferences table).
    """
    if file is None:
        raise HTTPException(400, "file is required")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(400, "Empty upload")

    # Decode image
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception:
        raise HTTPException(400, "Could not decode image")

    # Run detector
    try:
        detections = vision.detect_foods(img)
    except Exception as e:
        # If the model failed to load or run, surface a 500 so you see it in logs.
        raise HTTPException(500, f"vision_model_error: {e}")

    if not detections:
        # Log an empty inference but return no candidates
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inferences (image_url, detections, kb_hits)
                    VALUES (%s, %s::jsonb, %s::jsonb);
                    """,
                    (None, json.dumps([]), json.dumps([])),
                )
                conn.commit()
        except Exception:
            pass

        return {"candidates": []}

    # Deduplicate labels, keeping the highest model score per label
    best_det_by_label: Dict[str, Dict[str, Any]] = {}
    for d in detections:
        label = str(d.get("label") or "").strip()
        if not label:
            continue
        prev = best_det_by_label.get(label)
        score = float(d.get("score") or 0.0)
        if prev is None or score > float(prev.get("score") or 0.0):
            best_det_by_label[label] = d

    labels = list(best_det_by_label.keys())

    # Use your existing fuzzy resolver to map labels → foods
    hits = _resolve_tokens_against_db(labels)

    # Best DB hit per token
    best_hit_by_token: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        token = str(h.get("token") or "")
        if not token:
            continue
        prev = best_hit_by_token.get(token)
        score = float(h.get("db_score") or 0.0)
        if prev is None or score > float(prev.get("db_score") or 0.0):
            best_hit_by_token[token] = h

    candidates: List[Dict[str, Any]] = []
    items_for_overall: List[Dict[str, Any]] = []

    # Join detections + DB hits by label/token
    for label, det in best_det_by_label.items():
        hit = best_hit_by_token.get(label)
        if not hit:
            # For now, skip detections that don't map into the KB
            continue

        status = hit.get("status")

        cand = {
            # Used by the mobile app mapping in src/api.ts
            "label": label,
            "model_label": label,
            "food_id": hit.get("food_id"),
            "name": hit.get("name") or label,
            "status": status,
            "matched_from": hit.get("matched_from"),
            "model_score": float(det.get("score") or 0.0),
            "det_conf": float(det.get("score") or 0.0),
            "db_score": hit.get("db_score"),
            "sources": hit.get("sources") or [],
            "notes": hit.get("notes"),
        }

        candidates.append(cand)
        items_for_overall.append({"status": status})

    overall_status = _pick_overall_status(items_for_overall)

    # Log to inferences table for auditing
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO inferences (image_url, detections, final_status, kb_hits)
                VALUES (%s, %s::jsonb, %s::food_status, %s::jsonb);
                """,
                (
                    None,  # later: S3 URL or similar
                    json.dumps(detections),
                    overall_status,
                    json.dumps(hits),
                ),
            )
            conn.commit()
    except Exception:
        # Non-fatal: don't break the API if logging fails
        pass

    return {"candidates": candidates, "overall_status": overall_status}



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
# Barcode → OpenFoodFacts → ingredients → hits
# -------------------------------------------------

OPENFOODFACTS_PRODUCT_URL = os.getenv(
    "OPENFOODFACTS_PRODUCT_URL",
    "https://world.openfoodfacts.org/api/v0/product/{barcode}.json",
)


def _fetch_openfoodfacts_product(barcode: str) -> Optional[Dict[str, Any]]:
    """
    Look up a product in OpenFoodFacts by barcode.

    Returns a dict with keys: name, ingredients_text
    or None if the product cannot be found / parsed.
    """
    try:
        url = OPENFOODFACTS_PRODUCT_URL.format(barcode=barcode)
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # network / JSON / etc.
        print("WARN: OpenFoodFacts lookup failed:", exc)
        return None

    if not isinstance(data, dict) or data.get("status") != 1:
        # 0 = product not found in OFF
        return None

    product = data.get("product") or {}
    if not isinstance(product, dict):
        return None

    # Product name
    name = (
        product.get("product_name")
        or product.get("generic_name")
        or "Scanned product"
    )

    # Ingredients text: prefer English, then generic, then build from list
    ingredients_text = ""
    for key in ("ingredients_text_en", "ingredients_text"):
        val = product.get(key)
        if isinstance(val, str) and val.strip():
            ingredients_text = val.strip()
            break

    if not ingredients_text:
        ing_list = product.get("ingredients")
        if isinstance(ing_list, list):
            parts: List[str] = []
            for ing in ing_list:
                if isinstance(ing, dict):
                    txt = ing.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            if parts:
                ingredients_text = ", ".join(parts)

    return {
        "name": str(name).strip() or "Scanned product",
        "ingredients_text": ingredients_text.strip(),
    }


@app.post("/barcode/resolve")
def barcode_resolve(payload: Dict[str, Any]):
    """
    Resolve a barcode to an overall product rating + ingredient ratings.

    Flow:
    - Require a barcode.
    - If ingredients_text is provided in payload, use it (handy for debugging).
    - Otherwise, call OpenFoodFacts to fetch product + ingredients.
    - If still nothing, return an "error" payload that the mobile app shows nicely.
    - Once we have ingredients_text, tokenize and match against foods+synonyms.
    """
    code = str(payload.get("barcode", "")).strip()
    if not code:
        raise HTTPException(400, "barcode is required")

    # Optional overrides coming from the client:
    ingredients_text = str(payload.get("ingredients_text") or "").strip()
    display_name = str(payload.get("display_name") or "").strip()

    # 1) If no ingredients passed in, try OpenFoodFacts
    if not ingredients_text:
        product = _fetch_openfoodfacts_product(code)

        if not product:
            # Let the client show a friendly "not found" bubble
            return {
                "barcode": code,
                "error": "barcode_not_found",
                "message": (
                    "This barcode is not in our food database. "
                    "It may be a non-food item or a product we haven't indexed yet."
                ),
            }

        ingredients_text = product.get("ingredients_text", "").strip()
        if not ingredients_text:
            return {
                "barcode": code,
                "error": "ingredients_missing",
                "message": (
                    "We found this product, but it does not have a readable "
                    "ingredients list yet."
                ),
            }

        if not display_name:
            display_name = product.get("name") or "Scanned product"

    # Final safety: if somehow still no ingredients, bail gracefully
    if not ingredients_text:
        return {
            "barcode": code,
            "error": "ingredients_missing",
            "message": (
                "We couldn't find ingredients for this product yet. "
                "Try typing the ingredient names manually."
            ),
        }

    # 2) Now we have ingredients: resolve against foods + synonyms
    tokens = _tokenize_ingredients(ingredients_text)
    hits = _resolve_tokens_against_db(tokens)

    return {
        "barcode": code,
        "display_name": display_name or "Scanned product",
        "raw_ingredients": ingredients_text,
        "hits": hits,
        "overall_status": _pick_overall_status(hits),
    }
