-- db/06_constraints.sql

-- 1) foods: unique canonical_name (case-insensitive)
CREATE UNIQUE INDEX IF NOT EXISTS ux_foods_canonical_name_lower
  ON foods (lower(canonical_name));

-- 2) synonyms: unique per food (case-insensitive)
CREATE UNIQUE INDEX IF NOT EXISTS ux_synonyms_food_name_lower
  ON synonyms (food_id, lower(name));

-- 3) prevent: synonym text == another food's canonical_name
CREATE OR REPLACE FUNCTION forbid_synonym_food_conflict()
RETURNS trigger AS $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM foods f
    WHERE lower(f.canonical_name) = lower(NEW.name)
      AND f.id <> NEW.food_id
  ) THEN
    RAISE EXCEPTION 'Synonym "%" conflicts with another food''s canonical_name', NEW.name;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_synonym_food_conflict ON synonyms;
CREATE TRIGGER trg_synonym_food_conflict
  BEFORE INSERT OR UPDATE ON synonyms
  FOR EACH ROW
  EXECUTE FUNCTION forbid_synonym_food_conflict();

-- 4) status whitelist (adjust if you add UNKNOWN later)
ALTER TABLE foods
  ALTER COLUMN canonical_name SET NOT NULL,
  ALTER COLUMN default_status SET NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.constraint_column_usage
    WHERE table_name='foods' AND constraint_name='chk_food_status'
  ) THEN
    ALTER TABLE foods
      ADD CONSTRAINT chk_food_status
      CHECK (default_status IN ('SAFE','CAUTION','UNSAFE'));
  END IF;
END$$;
