# api/main.py
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dogfood")
app = FastAPI(title="NibbleCheck API", version="0.1.0")

def db():
    return psycopg.connect(DATABASE_URL, autocommit=True)

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
            } for r in rows
        ]
    }

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
