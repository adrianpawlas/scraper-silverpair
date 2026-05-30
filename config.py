"""
Configuration module for the Silverpair scraper.

Sensitive values (Supabase creds) are read from environment variables first,
with hardcoded defaults as fallback for local development.

In production (GitHub Actions), set:
  - SUPABASE_URL
  - SUPABASE_ANON_KEY
as repository secrets.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Scraper Settings ---
SOURCE_NAME = "scraper-silverpair"
BRAND_NAME = "Silverpair"

BASE_URL = "https://silverpair.co"
COLLECTIONS_URL = f"{BASE_URL}/collections/all"

# --- Embedding Model ---
EMBEDDING_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
# Device: override via env var (e.g., "cpu" on GitHub Actions).
# Defaults to "mps" for local Apple Silicon dev.
DEVICE = os.getenv("DEVICE", "mps")

# --- Supabase ---
# In production (GitHub Actions), pass these as secrets via environment variables.
# Local: falls back to the hardcoded values below.
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SUPABASE_TABLE = "products"

# --- Request Settings ---
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.0  # seconds between page requests to be respectful

# --- Batch & Upsert Settings ---
BATCH_SIZE = 50  # products per batch insert
BATCH_RETRIES = 3  # retry attempts for failed batch inserts

# --- Embedding Settings ---
EMBEDDING_STAGGER_DELAY = 0.5  # seconds between HuggingFace API calls

# --- Stale Product Cleanup ---
# Products not seen for this many days (2 weekly runs) are auto-deleted
STALE_DAYS_THRESHOLD = 14
