-- db/05_functions.sql

-- (safety) require pg_trgm already created in 00_extensions.sql

-- Quick helper indexes if not already present
CREATE INDEX IF NOT EXISTS ix_synonyms_food_id ON synonyms(food_id);
CREATE INDEX IF NOT EXISTS ix_rules_food_id    ON rules(food_id);

-- Basic label resolver (per-match, not deduped)
CREATE OR REPLACE FUNCTION resolve_label(q text, limit_n int DEFAULT 20)
RETURNS TABLE (
  food_id int,
  matched text,
  matched_from text,
  score real
)
LANGUAGE sql
STABLE
AS $$
  SELECT t.food_id, t.matched, t.matched_from, t.score
  FROM (
    SELECT
      f.id                            AS food_id,
      f.canonical_name                AS matched,
      'food'::text                    AS matched_from,
      similarity(f.canonical_name, q) AS score
    FROM foods f
    WHERE f.canonical_name % q

    UNION ALL

    SELECT
      s.food_id                       AS food_id,
      s.name                          AS matched,
      'synonym'::text                 AS matched_from,
      similarity(s.name, q)           AS score
    FROM synonyms s
    WHERE s.name % q
  ) AS t
  ORDER BY t.score DESC
  LIMIT limit_n;
$$;

-- Enriched search (one best row per food_id)
CREATE OR REPLACE FUNCTION search_foods_enriched(q text, limit_n int DEFAULT 20)
RETURNS TABLE(
  food_id int,
  canonical_name text,
  group_name text,
  default_status text,
  matched text,
  matched_from text,
  score real
)
LANGUAGE sql
STABLE
AS $$
  SELECT DISTINCT ON (t.food_id)
    t.food_id,
    f.canonical_name,
    f.group_name,
    f.default_status,
    t.matched,
    t.matched_from,
    t.score
  FROM (
    SELECT
      f.id                            AS food_id,
      f.canonical_name                AS matched,
      'food'::text                    AS matched_from,
      similarity(f.canonical_name, q) AS score
    FROM foods f
    WHERE f.canonical_name % q

    UNION ALL

    SELECT
      s.food_id                       AS food_id,
      s.name                          AS matched,
      'synonym'::text                 AS matched_from,
      similarity(s.name, q)           AS score
    FROM synonyms s
    WHERE s.name % q
  ) t
  JOIN foods f ON f.id = t.food_id
  ORDER BY t.food_id, t.score DESC
  LIMIT limit_n;
$$;
