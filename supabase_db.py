"""
Supabase database module for inserting scraped products into the products table.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client

import config

logger = logging.getLogger(__name__)


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

    def upsert_product(
        self,
        product: dict,
        image_embedding: Optional[list[float]] = None,
        info_embedding: Optional[list[float]] = None,
    ) -> bool:
        """
        Insert or update a product in the products table.
        
        Args:
            product: Scraped product data dict
            image_embedding: 768-dim image embedding vector
            info_embedding: 768-dim text embedding vector
        
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            raise RuntimeError("Not connected to Supabase. Call connect() first.")

        # Build the row data matching the table schema
        row = {
            "id": product["id"],
            "source": product["source"],
            "product_url": product["product_url"],
            "affiliate_url": product.get("affiliate_url"),
            "image_url": product["image_url"],
            "brand": product["brand"],
            "title": product["title"],
            "description": product.get("description"),
            "category": product.get("category"),
            "gender": product.get("gender"),
            "price": product.get("price"),
            "sale": product.get("sale"),
            "second_hand": product.get("second_hand", False),
            "size": product.get("size"),
            "additional_images": product.get("additional_images"),
            "metadata": product.get("metadata"),
            "tags": product.get("tags"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_embedding": image_embedding,
            "info_embedding": info_embedding,
        }

        # Remove None values (let DB use defaults)
        row = {k: v for k, v in row.items() if v is not None}

        try:
            # Use upsert with the unique constraint (source, product_url)
            result = self.client.table(config.SUPABASE_TABLE).upsert(
                row,
                on_conflict="source, product_url",
            ).execute()

            if result.data:
                logger.info(f"Successfully upserted product: {product['title']}")
                return True
            else:
                logger.warning(f"No data returned for product: {product['title']}")
                return False

        except Exception as e:
            logger.error(f"Failed to upsert product {product['title']}: {e}")
            return False

    def get_existing_product_ids(self) -> set[str]:
        """
        Get all existing product IDs from the database.
        Used to check which products have already been processed.
        """
        if self.client is None:
            raise RuntimeError("Not connected to Supabase. Call connect() first.")

        try:
            result = self.client.table(config.SUPABASE_TABLE).select("id").execute()
            return {row["id"] for row in (result.data or [])}
        except Exception as e:
            logger.error(f"Failed to fetch existing products: {e}")
            return set()
