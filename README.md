# NibbleCheck 🐶🍽️

**NibbleCheck** helps you decide whether a food in a photo is **SAFE**, **CAUTION**, or **UNSAFE** for dogs — and *why*.  
It combines computer vision to recognize foods with a curated knowledge base (rules, synonyms, canonical items) so decisions are accurate and explainable.

---

## What NibbleCheck Does

- **Photo → Verdict**  
  Upload a photo of food(s). The app detects items, normalizes names (e.g., “grapes”, “grape jelly”, “dark choclate” typo), and returns a per-item verdict.

- **Explains the Why**  
  Every result includes a rule-backed rationale, e.g.,  
  **UNSAFE** — contains xylitol,  
  **CAUTION** — apple **seeds** are unsafe; flesh is okay,  
  **SAFE** — plain cooked chicken without bones/skin.

- **Handles Real-World Variants**  
  Synonyms, plurals, brands, misspellings, and prep details (raw/cooked, pits/seeds/bones, oil/seasoning).

- **Learns Over Time**  
  User feedback and edge cases are logged for quality improvements.

---

## How It Works (End-to-End)

1. **Vision**  
   A lightweight image model detects candidate foods from the photo (single or multi-item). Detections include labels and confidences.

2. **Normalization**  
   Detected text is mapped to canonical foods via:
   - exact lookup on **synonyms**
   - exact lookup on **canonical name**
   - fuzzy match (Postgres **pg_trgm**) on canonical names

3. **Rule Reasoning**  
   Structured rules (JSON) adjust or explain the default status for each food and preparation context.  
   Examples: `{ "part":"seeds" }`, `{ "prepared":"raw" }`, `{ "contains":"xylitol" }`.

4. **Verdict & Explanation**  
   For each item, return a final status (**SAFE / CAUTION / UNSAFE**) plus a short rationale and the matched knowledge sources.

5. **Feedback Loop**  
   Results and optional user feedback are stored (no PII) to improve synonyms, rules, and model performance.

---

## Major Components

- **Mobile/Web App**  
  Simple, fast UI to take/upload a photo, review detections, and see verdicts + explanations.

- **API Layer**  
  Minimal FastAPI service exposing:
  - `POST /classify` — image in, structured verdicts out  
  - `GET /resolve?label=...` — resolve a text label to a verdict + explanation  
  Handles synonym/canonical/fuzzy matching, rule evaluation, and logging.

- **Knowledge Base (PostgreSQL)**  
  - `foods` — canonical entries with default status  
  - `synonyms` — robust mapping for variants/plurals/misspellings  
  - `rules` — JSON conditions for parts/prep/ingredients  
  - `inferences` — optional logging of detections/results/feedback  
  Trigram indexes (`pg_trgm`) provide fast fuzzy matching.

---

## Tech Stack

- **Computer Vision:** lightweight classifier/detector (exportable for server or on-device)  
- **Backend:** FastAPI (Python), `psycopg` + pooling  
- **Database:** PostgreSQL (16–18), `pg_trgm`, enum types for statuses  
- **App:** React/React Native (or Flutter) for a mobile-first UI  
- **DevOps:** GitHub Actions CI, Docker (API/DB), environment-based config  
- **Security/Privacy:** minimal data retention, no PII required, opt-in analytics