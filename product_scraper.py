"""
Product scraper for silverpair.co.
Extracts product data from the Shopify store using embedded JSON data.
"""

import re
import json
import time
import logging
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


def get_page_html(url: str) -> str:
    """Fetch HTML content from a URL with retry logic."""
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                timeout=config.REQUEST_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def extract_product_urls() -> list[str]:
    """
    Scrape the collections page and extract all product URLs.
    Uses the Shopify product grid links from the HTML.
    """
    logger.info(f"Fetching collections page: {config.COLLECTIONS_URL}")
    html = get_page_html(config.COLLECTIONS_URL)

    # Extract all product URLs from href attributes
    product_handles = set()
    # Pattern: /products/product-handle
    pattern = re.compile(r'/products/([a-zA-Z0-9][a-zA-Z0-9-]*)')
    for match in pattern.finditer(html):
        handle = match.group(1)
        product_handles.add(handle)

    # Sort for consistent ordering
    product_urls = [
        urljoin(config.BASE_URL, f"/products/{handle}")
        for handle in sorted(product_handles)
    ]

    logger.info(f"Found {len(product_urls)} products on collections page")
    return product_urls


def extract_meta_json(html: str) -> Optional[dict]:
    """Extract the Shopify `var meta = {...}` JSON from the page."""
    match = re.search(r'var\s+meta\s*=\s*(\{.*?\});', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse meta JSON")
    return None


def extract_product_group_json(html: str) -> Optional[dict]:
    """
    Extract the ProductGroup JSON-LD from the page.
    This contains category, description, brand, and full variant info.
    """
    # Try to find the ProductGroup schema (we specifically want ProductGroup for categories)
    pattern = re.compile(
        r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        content = match.group(1).strip()
        try:
            data = json.loads(content)
            if data.get("@type") == "ProductGroup":
                return data
        except json.JSONDecodeError:
            continue
    return None


def extract_all_images(html: str) -> list[str]:
    """
    Extract all product image URLs from the page.
    Looks for images in the Shopify product media/gallery and JSON data.
    """
    images = set()
    soup = BeautifulSoup(html, "lxml")

    # Find images in the product gallery/media section
    # Shopify typically uses elements with image URLs in src/srcset/data attributes
    for img in soup.select(
        'img[src*="/files/"], '
        'img[data-src*="/files/"], '
        'img[srcset*="/files/"]'
    ):
        src = img.get("src") or img.get("data-src") or ""
        if src and "/files/" in src:
            if src.startswith("//"):
                src = "https:" + src
            # Add with original params - Shopify handles URL resolution
            images.add(src)

    # Also find images in JSON data (variants, media arrays)
    json_img_pattern = re.compile(
        r'"(https?:)?//[^"]*cdn\.shopify\.com[^"]*\.(png|jpg|jpeg|webp)[^"]*"'
    )
    for match in json_img_pattern.finditer(html):
        url = match.group(0).strip('"')
        if url.startswith("//"):
            url = "https:" + url
        images.add(url)

    return sorted(images)


def extract_page_prices(html: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract prices from the page in multiple formats.
    
    Returns: (czk_price_raw, usd_price_raw)
    czk_price_raw: e.g. "1.670,00 Kč"
    usd_price_raw: e.g. "$79.00"
    """
    soup = BeautifulSoup(html, "lxml")
    czk_price = None
    usd_price = None

    # Try to find CZK price in visible page text
    # Look for patterns like "1.670,00 Kč" in the text
    czk_pattern = re.compile(r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*Kč')
    
    # Search in the page text (visible text only)
    body_text = soup.get_text()
    czk_match = czk_pattern.search(body_text)
    if czk_match:
        czk_price = czk_match.group(0).strip()

    # Try OG meta tags for USD price
    og_price = soup.select_one('meta[property="og:price:amount"]')
    og_currency = soup.select_one('meta[property="og:price:currency"]')
    if og_price and og_price.get("content"):
        currency = og_currency.get("content", "USD") if og_currency else "USD"
        usd_price = f"${og_price['content']} {currency}"

    return czk_price, usd_price


def scrape_product(url: str) -> dict:
    """
    Scrape a single product page and extract all available data.
    
    Returns a dict with all product fields ready for database insertion.
    """
    logger.info(f"Scraping product: {url}")
    html = get_page_html(url)

    # Extract handle from URL
    handle = url.split("/products/")[-1].split("?")[0]

    # Get the meta JSON for variant/pricing data
    meta = extract_meta_json(html)
    product_data = meta.get("product", {}) if meta else {}

    # Get ProductGroup JSON-LD for category, description, etc.
    product_group = extract_product_group_json(html)

    # --- Extract fields ---

    # Title
    title = product_group.get("name") if product_group else ""
    if not title:
        # Fallback: extract from <title> tag or og:title
        title_match = re.search(r'<title>(.*?)\s*-\s*Silverpair</title>', html, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
        elif product_data:
            title = product_data.get("handle", "").replace("-", " ").title()

    # Description
    description = ""
    if product_group:
        desc = product_group.get("description", "")
        # Clean up the description (remove size chart HTML)
        if desc:
            # Extract first meaningful paragraph before "Size Chart"
            clean_desc = re.split(r'\s*Size\s*Chart\s*', desc, flags=re.IGNORECASE)[0].strip()
            # Clean HTML tags
            clean_desc = re.sub(r'<[^>]+>', ' ', clean_desc)
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
            description = clean_desc

    # Category - handle compound categories (e.g., "Sweaters & Hoodies" -> "Sweaters, Hoodies")
    category = ""
    if product_group:
        raw_category = product_group.get("category", "")
        if raw_category:
            # Split on "&" or "," and clean up
            parts = re.split(r'\s*[&,]\s*', raw_category)
            category = ", ".join(p.strip() for p in parts if p.strip())

    # Price (from JSON - primary: USD)
    prices_usd = set()
    czk_display_price = None
    
    if product_group and "hasVariant" in product_group:
        for variant in product_group["hasVariant"]:
            offers = variant.get("offers", {})
            price = offers.get("price")
            currency = offers.get("priceCurrency", "USD")
            if price and currency == "USD":
                prices_usd.add(f"{price}{currency}")
    
    # Also get prices from page display (CZK)
    page_czk, page_usd = extract_page_prices(html)
    if page_czk:
        # Clean CZK format: "1.670,00 Kč" -> "1670.00CZK"
        czk_clean = page_czk.replace(".", "").replace(",", ".")
        # Remove non-numeric suffix
        czk_amount = re.search(r'[\d.]+', czk_clean)
        if czk_amount:
            czk_display_price = f"{czk_amount.group()}CZK"

    # Also add prices from meta JSON (in cents)
    if product_data and "variants" in product_data:
        for v in product_data["variants"]:
            price_cents = v.get("price")
            if price_cents:
                # Convert cents to dollars
                price_dollars = float(price_cents) / 100
                prices_usd.add(f"{price_dollars:.2f}USD")

    # Build price string
    price_parts = list(prices_usd)
    if czk_display_price:
        # Check if it's a different value than USD conversion
        price_parts.append(czk_display_price)
    price_str = ", ".join(sorted(price_parts)) if price_parts else None

    # Sale price - check for compare-at price
    sale_price = None
    if product_data and "variants" in product_data:
        for v in product_data["variants"]:
            compare_at = v.get("compare_at_price")
            if compare_at and compare_at > 0:
                sale_dollars = float(v.get("price", 0)) / 100
                # Format sale with CZK too
                sale_price = f"{sale_dollars:.2f}USD"
                if czk_display_price:
                    # Estimate CZK sale price (proportional)
                    sale_price += f" , {czk_display_price}"
                break

    # Images
    all_images = extract_all_images(html)
    
    # Also get images from variant data in ProductGroup
    variant_images = set()
    if product_group and "hasVariant" in product_group:
        for variant in product_group["hasVariant"]:
            img = variant.get("image", "")
            if img:
                variant_images.add(img)
    
    # Combine all unique images
    all_image_urls = list(variant_images) + [img for img in all_images if img not in variant_images]
    # Deduplicate while preserving order
    seen = set()
    unique_images = []
    for img in all_image_urls:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    
    main_image = unique_images[0] if unique_images else ""
    additional_images = " , ".join(unique_images[1:]) if len(unique_images) > 1 else None

    # Gender - these are streetwear products, most are unisex
    gender = "unisex"
    # If it's clearly gendered (e.g., the brand only sells unisex streetwear)
    # Check if any specific indicators
    title_lower = title.lower()
    if any(w in title_lower for w in ["women", "woman", "female", "girls"]):
        gender = "women"
    elif any(w in title_lower for w in ["men", "man", "male", "boys"]):
        gender = "men"

    # Sizes & Colors - extract from variant names
    sizes = set()
    colors = set()
    if product_data and "variants" in product_data:
        for v in product_data["variants"]:
            name = v.get("public_title") or ""
            if not name:
                continue
            # Split by " / " and get parts
            parts = name.split(" / ")
            if len(parts) >= 2:
                size = parts[-1].strip()
                if size:
                    sizes.add(size)
            if len(parts) >= 1:
                color = parts[0].strip()
                if color:
                    colors.add(color)

    # Tags - extract from the product
    tags = []
    if product_group:
        pg_category = product_group.get("category", "")
        if pg_category:
            # Split compound categories like "Sweaters & Hoodies"
            for part in re.split(r'\s*[&,]\s*', pg_category):
                part = part.strip()
                if part:
                    tags.append(part)
    
    # Build metadata - all info in one JSON string
    metadata = {
        "title": title,
        "description": description,
        "category": category,
        "gender": gender,
        "price": price_str,
        "sale": sale_price,
        "sizes": sorted(sizes) if sizes else [],
        "colors": sorted(colors) if colors else [],
        "tags": tags,
        "handle": handle,
        "shopify_id": str(product_data.get("id", "")),
        "vendor": product_data.get("vendor", ""),
    }
    if product_group:
        metadata["product_group_id"] = product_group.get("productGroupID", "")

    # Build comprehensive text for info_embedding (includes everything about the product)
    info_text_parts = [
        f"Title: {title}",
        f"Brand: {config.BRAND_NAME}",
        f"Description: {description}",
        f"Category: {category}" if category else "",
        f"Gender: {gender}",
        f"Price: {price_str}" if price_str else "",
        f"Sale: {sale_price}" if sale_price else "",
        f"Sizes: {', '.join(sorted(sizes))}" if sizes else "",
        f"Colors: {', '.join(sorted(colors))}" if colors else "",
        f"Tags: {', '.join(tags)}" if tags else "",
    ]
    info_text = ". ".join(p for p in info_text_parts if p)

    # Build the result dict
    result = {
        "id": f"silverpair-{handle}",
        "source": config.SOURCE_NAME,
        "product_url": url,
        "affiliate_url": None,
        "image_url": main_image,
        "brand": config.BRAND_NAME,
        "title": title,
        "description": description,
        "category": category if category else None,
        "gender": gender,
        "price": price_str,
        "sale": sale_price,
        "second_hand": False,
        "size": ", ".join(sorted(sizes)) if sizes else None,
        "additional_images": additional_images,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "tags": tags if tags else None,
        "info_text": info_text,  # For generating info_embedding
    }

    return result
