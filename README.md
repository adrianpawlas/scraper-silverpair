# Silverpair Scraper

Scrapes all products from [silverpair.co](https://silverpair.co), generates image and text embeddings using [google/siglip-base-patch16-384](https://huggingface.co/google/siglip-base-patch16-384) (768-dim), and imports everything to a Supabase database.

## Usage

```bash
# Full scrape (all products → embeddings → Supabase)
python scraper.py

# Dry run (scrape only, no database)
python scraper.py --dry-run

# Resume mode (skip already-imported products)
python scraper.py --resume

# Scrape a single product
python scraper.py --product baggy-fit-jeans

# Skip embedding generation (faster)
python scraper.py --skip-embeddings
```

## Automation

A GitHub Actions workflow runs the scraper **every Thursday at ~14:00 Czech time** (12:00 UTC — note this is 14:00 CEST in summer, 13:00 CET in winter).
You can also trigger it manually from the GitHub UI under **Actions → Silverpair Scraper → Run workflow**.

### Required GitHub Secrets

Set these in your repo **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | Your Supabase anon/public key |

## Local Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # Optional: override defaults
python scraper.py
```

## Tech Stack

- **Scraper**: `requests` + `BeautifulSoup` + Shopify JSON-LD parsing (no headless browser)
- **Embeddings**: `google/siglip-base-patch16-384` via HuggingFace `transformers` (768-dim image + text embeddings)
- **Database**: Supabase (upsert with `source, product_url` conflict resolution)
- **Acceleration**: Apple Silicon MPS (local) / CPU (GitHub Actions)
