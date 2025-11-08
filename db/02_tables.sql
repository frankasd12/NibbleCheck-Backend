-- foods (unchanged)
CREATE TABLE IF NOT EXISTS foods (
  id             SERIAL PRIMARY KEY,
  canonical_name TEXT NOT NULL,
  group_name     TEXT,
  default_status food_status NOT NULL,
  notes          TEXT,
  sources        TEXT[],
  created_at     TIMESTAMPTZ DEFAULT now(),
  updated_at     TIMESTAMPTZ DEFAULT now()
);

-- add timestamps to synonyms
CREATE TABLE IF NOT EXISTS synonyms (
  id         SERIAL PRIMARY KEY,
  food_id    INTEGER NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- add timestamps to rules
CREATE TABLE IF NOT EXISTS rules (
  id         SERIAL PRIMARY KEY,
  food_id    INTEGER NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
  condition  JSONB NOT NULL,
  status     food_status NOT NULL,
  rationale  TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- align inferences with your DB (uses uploaded_at)
CREATE TABLE IF NOT EXISTS inferences (
  id            SERIAL PRIMARY KEY,
  uploaded_at   TIMESTAMPTZ DEFAULT now(),
  image_url     TEXT,
  detections    JSONB,
  final_status  food_status,
  kb_hits       JSONB,
  user_feedback JSONB
);

-- updated_at helper (unchanged)
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- triggers for all tables that have updated_at
DROP TRIGGER IF EXISTS trg_foods_updated_at ON foods;
CREATE TRIGGER trg_foods_updated_at
BEFORE UPDATE ON foods
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_synonyms_updated_at ON synonyms;
CREATE TRIGGER trg_synonyms_updated_at
BEFORE UPDATE ON synonyms
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_rules_updated_at ON rules;
CREATE TRIGGER trg_rules_updated_at
BEFORE UPDATE ON rules
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
