#!/usr/bin/env python3
"""
Silverpair Product Scraper
===========================
Full scraper that:
1. Scrapes all products from silverpair.co
2. Downloads product images
3. Generates 768-dim image & text embeddings using google/siglip-base-patch16-384
4. Inserts everything into Supabase products table

Usage:
    python scraper.py [--skip-embeddings] [--dry-run]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

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
        "--resume",
        action="store_true",
        help="Skip products that already exist in the database",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Silverpair Product Scraper")
    logger.info(f"Model: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}-dim)")
    logger.info(f"Source: {config.SOURCE_NAME}")
    logger.info(f"Brand: {config.BRAND_NAME}")
    logger.info("=" * 60)

    # --- Step 1: Get product URLs ---
    if args.product:
        if args.product.startswith("http"):
            product_urls = [args.product]
        else:
            product_urls = [f"{config.BASE_URL}/products/{args.product}"]
        logger.info(f"Scraping single product: {product_urls[0]}")
    else:
        logger.info("Fetching product list from collections page...")
        product_urls = extract_product_urls()
        logger.info(f"Found {len(product_urls)} products")

    if not product_urls:
        logger.error("No products found. Exiting.")
        sys.exit(1)

    # --- Step 2: Set up database connection ---
    db = SupabaseClient()
    existing_ids = set()
    if not args.dry_run:
        try:
            db.connect()
            if args.resume:
                existing_ids = db.get_existing_product_ids()
                logger.info(f"Found {len(existing_ids)} existing products in database")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            if not args.dry_run:
                sys.exit(1)

    # --- Step 3: Initialize embedder (if needed) ---
    embedder = None
    if not args.skip_embeddings:
        logger.info("Initializing SigLIP embedding model...")
        try:
            from embeddings import SigLIPEmbedder

            embedder = SigLIPEmbedder()
            embedder.load_model()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Continuing without embeddings...")
            embedder = None

    # --- Step 4: Scrape each product ---
    results = {"success": 0, "skipped": 0, "failed": 0}

    for i, url in enumerate(product_urls, 1):
        handle = url.split("/products/")[-1].split("?")[0]
        product_id = f"silverpair-{handle}"

        # Skip if already exists (resume mode)
        if args.resume and product_id in existing_ids:
            logger.info(f"[{i}/{len(product_urls)}] Skipping (already exists): {handle}")
            results["skipped"] += 1
            continue

        logger.info(f"[{i}/{len(product_urls)}] Scraping: {handle}")

        try:
            # Scrape the product
            product = scrape_product(url)

            if args.dry_run:
                print(f"\n--- {product['title']} ---")
                for key, value in product.items():
                    if key == "info_text":
                        continue
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    print(f"  {key}: {value}")
                results["success"] += 1
                continue

            # Generate embeddings
            image_embedding = None
            info_embedding = None

            if embedder and product.get("image_url"):
                logger.info(f"  Generating image embedding...")
                image_embedding = embedder.embed_image_from_url(product["image_url"])
                if image_embedding:
                    logger.info(f"  Image embedding: {len(image_embedding)} dims")
                else:
                    logger.warning(f"  Failed to generate image embedding")

            if embedder and product.get("info_text"):
                logger.info(f"  Generating text embedding...")
                try:
                    info_embedding = embedder.embed_text(product["info_text"])
                    if info_embedding:
                        logger.info(f"  Text embedding: {len(info_embedding)} dims")
                except Exception as e:
                    logger.warning(f"  Failed to generate text embedding: {e}")

            # Insert into Supabase
            success = db.upsert_product(
                product,
                image_embedding=image_embedding,
                info_embedding=info_embedding,
            )

            if success:
                results["success"] += 1
                logger.info(f"  ✓ Imported: {product['title']}")
            else:
                results["failed"] += 1
                logger.error(f"  ✗ Failed to import: {product['title']}")

        except Exception as e:
            results["failed"] += 1
            logger.error(f"  ✗ Error processing {handle}: {e}", exc_info=True)

        # Be respectful with a small delay
        if i < len(product_urls):
            time.sleep(config.REQUEST_DELAY)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info(f"  Successful: {results['success']}")
    logger.info(f"  Skipped:    {results['skipped']}")
    logger.info(f"  Failed:     {results['failed']}")
    logger.info("=" * 60)

    # Cleanup
    if embedder:
        embedder.cleanup()

    # Save results to file for reference
    if not args.dry_run:
        result_path = Path("scraper_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
