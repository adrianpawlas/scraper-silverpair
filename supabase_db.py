"""
Supabase database module for the Silverpair scraper.

Provides:
- Batch upsert (50 products at a time) with retry logic
- Smart field-level comparison to detect changes
- Stale product detection and cleanup
- Failed-batch logging
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from supabase import create_client, Client

import config

logger = logging.getLogger(__name__)

FAILED_BATCH_LOG = Path("failed_batch_inserts.jsonl")


class SupabaseClient:
    """Handles all Supabase database operations."""

    def __init__(self):
        self.client: Optional[Client] = None

    def connect(self):
        """Connect to Supabase."""
        logger.info("Connecting to Supabase...")
        self.client = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_ANON_KEY,
        )
        logger.info("Connected to Supabase")

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_source_products(self) -> dict[str, dict]:
        """
        Fetch ALL existing products for this source from the database.

        Returns a dict keyed by *product_url*, with each value being the
        full database row (including parsed metadata under the key
        ``"_parsed_metadata"``).
        """
        if self.client is None:
            raise RuntimeError("Not connected to Supabase. Call connect() first.")

        logger.info("Fetching existing products from Supabase...")

        try:
            result = (
                self.client
                .table(config.SUPABASE_TABLE)
                .select("*")
                .eq("source", config.SOURCE_NAME)
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to fetch source products: {e}")
            return {}

        rows: list[dict] = result.data or []
        products: dict[str, dict] = {}

        for row in rows:
            # Parse the metadata JSON string once so callers don't have to
            raw = row.get("metadata") or "{}"
            try:
                row["_parsed_metadata"] = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                row["_parsed_metadata"] = {}

            product_url = row.get("product_url")
            if product_url:
                products[product_url] = row

        logger.info(f"Found {len(products)} existing products for source '{config.SOURCE_NAME}'")
        return products

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(value):
        """Normalise a value for comparison (None / empty string → None)."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        # Lists are stored as arrays in Postgres; compare them as tuples for
        # stability (sorted).
        if isinstance(value, list):
            return tuple(value)
        return value

    def compare_product_changed(self, scraped: dict, existing: dict) -> bool:
        """
        Return ``True`` if any meaningful field differs between the
        freshly-scraped product and the existing database record.

        Fields checked: title, description, category, price, sale, size,
        tags, image_url, additional_images.
        """
        fields = [
            "title",
            "description",
            "category",
            "price",
            "sale",
            "size",
            "tags",
            "image_url",
            "additional_images",
        ]
        for field in fields:
            a = self._normalise(scraped.get(field))
            b = self._normalise(existing.get(field))
            if a != b:
                logger.debug(
                    "    change detected in '%s': %r -> %r", field, b, a
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Batch upsert
    # ------------------------------------------------------------------

    def build_db_row(
        self,
        scraped: dict,
        image_embedding: Optional[list[float]] = None,
        info_embedding: Optional[list[float]] = None,
    ) -> dict:
        """
        Build a complete database row from scraped product data.

        Injects ``last_seen_at`` into the product's metadata and sets
        ``created_at`` to the current UTC timestamp.
        """
        # Parse existing metadata, inject last_seen_at
        metadata_str = scraped.get("metadata") or "{}"
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else {}
        except json.JSONDecodeError:
            metadata = {}

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata["last_seen_at"] = now_iso

        row = {
            "id": scraped["id"],
            "source": scraped["source"],
            "product_url": scraped["product_url"],
            "affiliate_url": scraped.get("affiliate_url"),
            "image_url": scraped["image_url"],
            "brand": scraped["brand"],
            "title": scraped["title"],
            "description": scraped.get("description"),
            "category": scraped.get("category"),
            "gender": scraped.get("gender"),
            "price": scraped.get("price"),
            "sale": scraped.get("sale"),
            "second_hand": scraped.get("second_hand", False),
            "size": scraped.get("size"),
            "additional_images": scraped.get("additional_images"),
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "tags": scraped.get("tags"),
            "created_at": now_iso,
            "image_embedding": image_embedding,
            "info_embedding": info_embedding,
        }

        # Strip keys whose value is None so Supabase uses DB defaults
        # (except for explicit boolean False / numeric 0).
        cleaned = {}
        for k, v in row.items():
            if v is not None:
                cleaned[k] = v
        return cleaned

    def batch_upsert(self, rows: list[dict]) -> dict:
        """
        Upsert products in batches of ``config.BATCH_SIZE``.

        Each batch is retried up to ``config.BATCH_RETRIES`` times.
        Products that still fail are appended to a local log file.

        Returns a dict::

            {"succeeded": int, "failed": list[dict]}
        """
        if self.client is None:
            raise RuntimeError("Not connected to Supabase. Call connect() first.")

        if not rows:
            return {"succeeded": 0, "failed": []}

        succeeded = 0
        failed_products: list[dict] = []

        # Split into chunks of BATCH_SIZE
        for chunk_start in range(0, len(rows), config.BATCH_SIZE):
            chunk = rows[chunk_start : chunk_start + config.BATCH_SIZE]
            batch_num = chunk_start // config.BATCH_SIZE + 1
            total_batches = (len(rows) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

            logger.info(
                "  Batch %d/%d (%d products) ...",
                batch_num, total_batches, len(chunk),
            )

            last_error = None
            for attempt in range(1, config.BATCH_RETRIES + 1):
                try:
                    result = (
                        self.client
                        .table(config.SUPABASE_TABLE)
                        .upsert(chunk, on_conflict="source, product_url")
                        .execute()
                    )

                    inserted = len(result.data) if result.data else 0
                    succeeded += inserted
                    logger.info(
                        "  ✓ Batch %d upserted %d products",
                        batch_num, inserted,
                    )
                    last_error = None
                    break

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "  ⚠ Batch %d attempt %d/%d failed: %s",
                        batch_num, attempt, config.BATCH_RETRIES, e,
                    )

            if last_error is not None:
                # All retries exhausted – log the failed products
                logger.error(
                    "  ✗ Batch %d failed after %d attempts. Logging %d products.",
                    batch_num, config.BATCH_RETRIES, len(chunk),
                )
                failed_products.extend(chunk)
                for product in chunk:
                    self._log_failed_product(product, str(last_error))

        return {
            "succeeded": succeeded,
            "failed": failed_products,
        }

    @staticmethod
    def _log_failed_product(product: dict, error: str):
        """Append a failed product to the local log file."""
        entry = {
            "id": product.get("id"),
            "product_url": product.get("product_url"),
            "title": product.get("title"),
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(FAILED_BATCH_LOG, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.error("Could not write to failed-batch log at %s", FAILED_BATCH_LOG)

    # ------------------------------------------------------------------
    # Stale product cleanup
    # ------------------------------------------------------------------

    def delete_stale_products(self, seen_product_urls: set[str]) -> int:
        """
        Delete products from this source that were **not** seen in the
        current scrape run and whose ``last_seen_at`` (stored in metadata)
        is older than ``config.STALE_DAYS_THRESHOLD`` days.

        Returns the number of products deleted.
        """
        if self.client is None:
            raise RuntimeError("Not connected to Supabase. Call connect() first.")

        logger.info("Checking for stale products to clean up ...")

        # 1. Fetch all products for this source
        all_products = self.fetch_source_products()

        # 2. Identify candidates — product_url not in current scrape
        candidates = [
            (url, row)
            for url, row in all_products.items()
            if url not in seen_product_urls
        ]

        if not candidates:
            logger.info("No stale candidates found — all products were seen.")
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=config.STALE_DAYS_THRESHOLD)
        to_delete: list[str] = []  # collect IDs

        for url, row in candidates:
            parsed = row.get("_parsed_metadata", {})
            last_seen_str = parsed.get("last_seen_at")

            if last_seen_str is None:
                # No last_seen_at → never tagged by this scraper.
                # Consider this product stale after the threshold, because
                # it was presumably imported by an older version that didn't
                # track this field, and it hasn't been seen since.
                last_seen = datetime.min.replace(tzinfo=timezone.utc)
            else:
                try:
                    last_seen = datetime.fromisoformat(last_seen_str)
                except (ValueError, TypeError):
                    last_seen = datetime.min.replace(tzinfo=timezone.utc)

            if last_seen < cutoff:
                product_id = row.get("id")
                if product_id:
                    to_delete.append(product_id)
                    logger.info(
                        "  Stale: %s  (last seen: %s)",
                        product_id, last_seen_str or "never",
                    )

        # 3. Delete in batches
        if not to_delete:
            logger.info("No products exceeded the stale threshold (no deletions needed).")
            return 0

        logger.info("Deleting %d stale products ...", len(to_delete))

        # Delete in small batches to avoid URL-length limits
        BATCH = 20
        deleted_count = 0
        for i in range(0, len(to_delete), BATCH):
            batch_ids = to_delete[i : i + BATCH]
            try:
                result = (
                    self.client
                    .table(config.SUPABASE_TABLE)
                    .delete()
                    .in_("id", batch_ids)
                    .execute()
                )
                deleted_count += len(result.data or [])
                logger.info("  Deleted batch of %d", len(batch_ids))
            except Exception as e:
                logger.error("  Failed to delete batch: %s", e)

        logger.info("Deleted %d stale products total.", deleted_count)
        return deleted_count
