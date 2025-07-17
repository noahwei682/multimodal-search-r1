from PIL import Image
import numpy as np
import os
import json
import aiohttp
import asyncio
from typing import List, Dict, Tuple, Optional
import hashlib
import pickle

# Cache configuration
CACHE_DIR = os.path.expanduser("~/.cache/mmsearch_r1/image_search")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(image_url: str) -> str:
    """Generate a unique cache file path for an image URL."""
    url_hash = hashlib.md5(image_url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.pkl")

def load_from_cache(image_url: str) -> Optional[Tuple[str, List[Image.Image], Dict]]:
    """Try to load search results from cache."""
    cache_path = get_cache_path(image_url)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading from cache: {e}")
    return None

def save_to_cache(image_url: str, results: Tuple[str, List[Image.Image], Dict]):
    """Save search results to cache."""
    cache_path = get_cache_path(image_url)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"Error saving to cache: {e}")

async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
    """Fetch and convert an image from URL to PIL Image."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                import io
                return Image.open(io.BytesIO(data))
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
    return None

async def perform_serpapi_search(image_url: str, api_key: str) -> List[Dict]:
    """Perform reverse image search using SerpAPI."""
    params = {
        "engine": "google_reverse_image",
        "image_url": image_url,
        "api_key": api_key,
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("https://serpapi.com/search", params=params) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    if "image_results" in results:
                        return results["image_results"][:5]  # Return top 5 results
                else:
                    print(f"SerpAPI error: {resp.status}")
        except Exception as e:
            print(f"Error performing SerpAPI search: {e}")
    return []

def call_image_search(image_url: str) -> Tuple[str, List[Image.Image], Dict]:
    """
    Perform image-based search using SerpAPI.
    
    Args:
        image_url (str): URL of the image to search for
        
    Returns:
        tool_returned_str (str): A string containing search results with titles
        tool_returned_images (List[PIL.Image.Image]): List of result thumbnails
        tool_stat (dict): Status information about the search
    """
    # Check cache first
    cached_results = load_from_cache(image_url)
    if cached_results:
        return cached_results

    # Get API key from environment
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    # Run async search
    results = asyncio.run(perform_serpapi_search(image_url, api_key))
    
    # Process results
    tool_returned_str = "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:\n"
    
    tool_returned_images = []
    
    # Fetch images asynchronously
    async def fetch_all_images():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for result in results:
                if "thumbnail" in result:
                    tasks.append(fetch_image(session, result["thumbnail"]))
            return await asyncio.gather(*tasks)
    
    images = asyncio.run(fetch_all_images())
    
    # Build response
    for i, (result, img) in enumerate(zip(results, images), 1):
        if img:
            tool_returned_images.append(img)
            title = result.get("title", "No title")
            tool_returned_str += f"{i}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
    
    # Status information
    tool_stat = {
        "success": len(tool_returned_images) > 0,
        "num_images": len(tool_returned_images),
    }
    
    # Cache the results
    results = (tool_returned_str, tool_returned_images, tool_stat)
    save_to_cache(image_url, results)
    
    return results
