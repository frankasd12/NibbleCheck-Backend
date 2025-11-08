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

-- 3) CAUTION/UNSAFE must have a source (adjust if sources stored elsewhere)
DO $$
DECLARE c int;
BEGIN
  SELECT COUNT(*)
  INTO c
  FROM foods
  WHERE default_status IN ('CAUTION','UNSAFE')
    AND (sources IS NULL OR sources = '' OR btrim(sources) = '');
  IF c > 0 THEN
    RAISE EXCEPTION 'Verify failed: % CAUTION/UNSAFE entries missing sources', c;
  END IF;
END$$;

-- 4) basic counts (visible in CI logs)
SELECT
  (SELECT COUNT(*) FROM foods)   AS foods,
  (SELECT COUNT(*) FROM synonyms) AS synonyms,
  (SELECT COUNT(*) FROM rules)    AS rules;
