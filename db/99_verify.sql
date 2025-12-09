-- db/99_verify.sql
\set ON_ERROR_STOP 1

-- 1) no dangling rules.food_id
DO $$
DECLARE c int;
BEGIN
  SELECT COUNT(*)
  INTO c
  FROM rules r LEFT JOIN foods f ON f.id = r.food_id
  WHERE f.id IS NULL;

  IF c > 0 THEN
    RAISE EXCEPTION 'Verify failed: % dangling rules.food_id', c;
  END IF;
END$$;

-- 2) no ambiguous synonyms (same text pointing to multiple foods)
DO $$
DECLARE c int;
BEGIN
  SELECT COUNT(*)
  INTO c
  FROM (
    SELECT lower(name) AS n, COUNT(DISTINCT food_id) AS k
    FROM synonyms
    GROUP BY lower(name)
    HAVING COUNT(DISTINCT food_id) > 1
  ) t;

  IF c > 0 THEN
    RAISE EXCEPTION 'Verify failed: % ambiguous synonym(s)', c;
  END IF;
END$$;

-- 3) CAUTION/UNSAFE must have at least one non-blank source (foods.sources is TEXT[])
DO $$
DECLARE c int;
BEGIN
  SELECT COUNT(*)
  INTO c
  FROM foods
  WHERE default_status IN ('CAUTION','UNSAFE')
    AND (
      -- sources is NULL or empty
      cardinality(coalesce(sources, ARRAY[]::text[])) = 0
      -- or all elements are blank after trimming
      OR NOT EXISTS (
        SELECT 1
        FROM unnest(coalesce(sources, ARRAY[]::text[])) AS s(val)
        WHERE btrim(val) <> ''
      )
    );

  IF c > 0 THEN
    RAISE WARNING 'Verify: % CAUTION/UNSAFE entries missing sources', c;
  END IF;
END$$;

-- 4) basic counts (visible in CI logs)
SELECT
  (SELECT COUNT(*) FROM foods)    AS foods,
  (SELECT COUNT(*) FROM synonyms) AS synonyms,
  (SELECT COUNT(*) FROM rules)    AS rules;
