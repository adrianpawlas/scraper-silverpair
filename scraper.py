#!/usr/bin/env python3
"""
Silverpair Product Scraper
===========================
Scrapes all products from silverpair.co, generates SigLIP 768-dim
image & text embeddings, and imports them into Supabase.

Features:
  - Batch inserts (50 per batch) with automatic retry on failure
  - Smart change detection — only re-processes products that actually changed
  - Stale product cleanup — deletes products not seen for 2 consecutive runs
  - Selective embedding generation — only for new or image-changed products
  - Staggered embedding calls (0.5 s between calls)
  - Comprehensive run summary (new / updated / unchanged / stale)

Usage:
    python scraper.py [--skip-embeddings] [--dry-run] [--product HANDLE]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from product_scraper import scrape_product, extract_product_urls
from supabase_db import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Silverpair products and import to Supabase"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip generating embeddings (faster, for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape products but don't insert into database",
    )
    parser.add_argument(
        "--product",
        type=str,
        help="Scrape a single product by URL or handle (e.g., 'baggy-fit-jeans')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-process all products (ignore change detection)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_timedelta(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class RunSummary:
    """Collects and prints a structured summary of the scraper run."""

    def __init__(self):
        self.new = 0
        self.updated = 0
        self.unchanged = 0
        self.skipped_failed = 0
        self.stale_deleted = 0
        self.embeddings_generated = 0
        self.start_time = time.time()

    @property
    def elapsed(self) -> str:
        return format_timedelta(time.time() - self.start_time)

    def print(self):
        print()
        logger.info("=" * 60)
        logger.info("RUN COMPLETE — %s", self.elapsed)
        logger.info("=" * 60)
        logger.info(f"  New products:        {self.new}")
        logger.info(f"  Updated products:    {self.updated}")
        logger.info(f"  Unchanged (skipped): {self.unchanged}")
        logger.info(f"  Failed:              {self.skipped_failed}")
        logger.info(f"  Stale deleted:       {self.stale_deleted}")
        logger.info(f"  Embeddings gen.:     {self.embeddings_generated}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    summary = RunSummary()

    logger.info("=" * 60)
    logger.info("Silverpair Product Scraper")
    logger.info(f"Model: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}-dim)")
    logger.info(f"Source: {config.SOURCE_NAME}")
    logger.info(f"Brand: {config.BRAND_NAME}")
    logger.info("=" * 60)

    # ---- Step 1: Get product URLs -----------------------------------------

    if args.product:
        if args.product.startswith("http"):
            product_urls = [args.product]
        else:
            product_urls = [f"{config.BASE_URL}/products/{args.product}"]
        logger.info("Scraping single product: %s", product_urls[0])
    else:
        logger.info("Fetching product list from collections page...")
        product_urls = extract_product_urls()
        logger.info("Found %d products", len(product_urls))

    if not product_urls:
        logger.error("No products found. Exiting.")
        sys.exit(1)

    # ---- Step 2: Connect to database & fetch existing products -------------

    db = SupabaseClient()
    existing_products: dict[str, dict] = {}

    if not args.dry_run:
        try:
            db.connect()
            existing_products = db.fetch_source_products()
        except Exception as e:
            logger.error("Failed to connect to Supabase: %s", e)
            if not args.dry_run:
                sys.exit(1)

    # ---- Step 3: Initialise embedder ---------------------------------------

    embedder = None
    if not args.skip_embeddings:
        logger.info("Initialising SigLIP embedding model...")
        try:
            from embeddings import SigLIPEmbedder

            embedder = SigLIPEmbedder()
            embedder.load_model()
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            logger.warning("Continuing without embeddings...")

    # ---- Step 4: Process each product --------------------------------------

    batch_rows: list[dict] = []           # collected for DB insert
    seen_product_urls: set[str] = set()   # tracks what we saw this run

    for i, url in enumerate(product_urls, 1):
        handle = url.split("/products/")[-1].split("?")[0]
        seen_product_urls.add(url)

        # --- 4a. Scrape ----------------------------------------------------
        logger.info("[%d/%d] Scraping: %s", i, len(product_urls), handle)

        try:
            product = scrape_product(url)
        except Exception as e:
            logger.error("  ✗ Failed to scrape %s: %s", handle, e)
            summary.skipped_failed += 1
            continue

        if args.dry_run:
            print(f"\n--- {product['title']} ---")
            for key, value in product.items():
                if key == "info_text":
                    continue
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"  {key}: {value}")
            summary.new += 1
            continue

        # --- 4b. Compare with existing -------------------------------------
        existing = existing_products.get(url)

        if existing and not args.force:
            changed = db.compare_product_changed(product, existing)

            # Force re-process if the existing product has a stale (or missing)
            # embedding version (stored in metadata) — this makes sure every
            # product gets the new fields (back_image_url, back_image_embedding,
            # embedding_version) when the embedding pipeline is updated.
            if not changed:
                existing_meta = existing.get("_parsed_metadata", {})
                existing_version = existing_meta.get("embedding_version")
                if existing_version is None or existing_version < config.EMBEDDING_VERSION:
                    logger.info(
                        "  → Stale embedding version (%s), re-processing",
                        existing_version,
                    )
                    changed = True

            if not changed:
                logger.info("  → Unchanged (skip)")
                summary.unchanged += 1
                continue
            else:
                logger.info("  → Changes detected")

        is_new = existing is None
        if is_new:
            logger.info("  → New product")

        # --- 4c. Generate embeddings ---------------------------------------
        image_embedding = None
        back_image_embedding = None
        info_embedding = None
        needs_embedding = is_new

        # Also regenerate if the image URL changed, or if the embedding
        # version is stale (ensures existing products get re-embedded when
        # the pipeline is updated).
        if not needs_embedding and existing:
            old_img = existing.get("image_url")
            new_img = product.get("image_url")
            old_back = existing.get("back_image_url")
            new_back = product.get("back_image_url")
            if old_img != new_img:
                logger.info("  → Front image URL changed, re-generating embeddings")
                needs_embedding = True
            elif old_back != new_back:
                logger.info("  → Back image URL changed, re-generating embeddings")
                needs_embedding = True
            else:
                existing_meta = existing.get("_parsed_metadata", {})
                existing_version = existing_meta.get("embedding_version")
                if existing_version is None or existing_version < config.EMBEDDING_VERSION:
                    logger.info(
                        "  → Stale embedding version (%s), re-generating embeddings",
                        existing_version,
                    )
                    needs_embedding = True

        if needs_embedding and embedder:
            # ============== Front image embedding ==============
            if product.get("image_url"):
                # Staggered call: small delay between *each* embedding gen
                # (skip delay for the very first product)
                if summary.embeddings_generated > 0:
                    time.sleep(config.EMBEDDING_STAGGER_DELAY)

                logger.info("  Generating front image embedding...")
                try:
                    image_embedding = embedder.embed_image_from_url(product["image_url"])
                    if image_embedding:
                        logger.info(
                            "  Front image embedding: %d dims ✓", len(image_embedding)
                        )
                        summary.embeddings_generated += 1
                except Exception as e:
                    logger.warning("  Failed to generate front image embedding: %s", e)

            # ============== Back image embedding ===============
            if product.get("back_image_url"):
                time.sleep(config.EMBEDDING_STAGGER_DELAY)

                logger.info("  Generating back image embedding...")
                try:
                    back_image_embedding = embedder.embed_image_from_url(product["back_image_url"])
                    if back_image_embedding:
                        logger.info(
                            "  Back image embedding: %d dims ✓", len(back_image_embedding)
                        )
                        summary.embeddings_generated += 1
                except Exception as e:
                    logger.warning("  Failed to generate back image embedding: %s", e)

            # ============== Text embedding ====================
            if product.get("info_text"):
                time.sleep(config.EMBEDDING_STAGGER_DELAY)

                logger.info("  Generating text embedding...")
                try:
                    info_embedding = embedder.embed_text(product["info_text"])
                    if info_embedding:
                        logger.info(
                            "  Text embedding: %d dims ✓", len(info_embedding)
                        )
                        summary.embeddings_generated += 1
                except Exception as e:
                    logger.warning("  Failed to generate text embedding: %s", e)

            # Tag embeddings with current version (only set when actually generated)
            product["embedding_version"] = config.EMBEDDING_VERSION
            if back_image_embedding is not None:
                product["back_image_embedding"] = back_image_embedding

        # --- 4d. Build DB row and collect into batch -----------------------

        row = db.build_db_row(
            product,
            image_embedding=image_embedding,
            info_embedding=info_embedding,
        )
        batch_rows.append(row)

        if is_new:
            summary.new += 1
        else:
            summary.updated += 1

        # Flush batch when full
        if len(batch_rows) >= config.BATCH_SIZE:
            result = db.batch_upsert(batch_rows)
            batch_rows = []
            if result["failed"]:
                summary.skipped_failed += len(result["failed"])

        # Be respectful to the source website between page fetches
        if i < len(product_urls):
            time.sleep(config.REQUEST_DELAY)

    # ---- Step 5: Flush remaining batch ------------------------------------
    if batch_rows:
        result = db.batch_upsert(batch_rows)
        if result["failed"]:
            summary.skipped_failed += len(result["failed"])

    # ---- Step 6: Clean up stale products ----------------------------------
    if not args.dry_run and not args.product:
        try:
            summary.stale_deleted = db.delete_stale_products(seen_product_urls)
        except Exception as e:
            logger.error("Stale-product cleanup failed: %s", e)

    # ---- Step 7: Print summary & save -------------------------------------
    summary.print()

    # Cleanup model memory
    if embedder:
        embedder.cleanup()

    if not args.dry_run:
        result_path = Path("scraper_results.json")
        with open(result_path, "w") as f:
            json.dump(
                {
                    "new": summary.new,
                    "updated": summary.updated,
                    "unchanged": summary.unchanged,
                    "failed": summary.skipped_failed,
                    "stale_deleted": summary.stale_deleted,
                    "embeddings_generated": summary.embeddings_generated,
                    "elapsed": summary.elapsed,
                    "run_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
