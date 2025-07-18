import asyncio
import json
from openai import OpenAI
import aiohttp

# Configuration Constants
OPENROUTER_API_KEY = "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"
SERPAPI_API_KEY = "7f39cccc407fb2e59ce52917d178354931cd5884062b2add658a9f1bf5943508"
JINA_API_KEY = "jina_c948193913304f68b5c6b68cf75e1987t-wKv5qheig7zcBRcrBDPmFDrER7"

# Endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SERPAPI_URL = "https://serpapi.com/search"
JINA_BASE_URL = "https://r.jina.ai/"

# Default LLM model
DEFAULT_MODEL = "qwen/qwen3-32b:free"

# Initialize OpenAI client
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/mshumer/OpenDeepResearcher",
        "X-Title": "OpenDeepResearcher"
    }
)

async def perform_search_async(query):
    """
    Asynchronously perform a Google search using SERPAPI for the given query.
    Returns a list of result URLs.
    """
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google"
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(SERPAPI_URL, params=params) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    if "organic_results" in results:
                        links = [item.get("link") for item in results["organic_results"] if "link" in item]
                        return links[:3]  # Limit to top 3 results
                    else:
                        print("No organic results in SERPAPI response.")
                        return []
                else:
                    text = await resp.text()
                    print(f"SERPAPI error: {resp.status} - {text}")
                    return []
        except Exception as e:
            print("Error performing SERPAPI search:", e)
            return []

async def fetch_webpage_text_async(url):
    """
    Asynchronously retrieve the text content of a webpage using Jina.
    """
    full_url = f"{JINA_BASE_URL}{url}"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(full_url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    text = await resp.text()
                    print(f"Jina fetch error for {url}: {resp.status} - {text}")
                    return ""
        except Exception as e:
            print("Error fetching webpage text with Jina:", e)
            return ""

async def summarize_text_async(text_query, page_text):
    """
    Use Qwen3-32B to generate a summary of the webpage content relevant to the query.
    """
    prompt = (
        "You are an expert information extractor. Given the user's query and webpage content, "
        "extract and summarize the most relevant information that helps answer the query. "
        "Be concise but comprehensive. Include only information that is directly relevant."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"Query: {text_query}\n\nWebpage Content (first 20000 chars):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            extra_body={}
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            print(f"Authentication failed: {error_msg}. Please check if your API keys are valid and have sufficient permissions.")
        elif "429" in error_msg:
            print(f"Rate limit exceeded: {error_msg}. Please check your API usage limits.")
        else:
            print(f"Error generating summary: {error_msg}")
        return ""

def call_text_search(text_query: str):
    """
    Perform text-based search using SerpAPI, JINA Reader, and Qwen3-32B for summarization.
    
    Args:
        text_query (str): The input query string used for text-based search.
    
    Returns:
        tool_returned_str (str): A string containing ranked search results with summaries.
        tool_stat (dict): A dictionary indicating tool execution status and metadata.
    """
    try:
        # Run the async search pipeline
        async def search_pipeline():
            # Get search results from SerpAPI
            links = await perform_search_async(text_query)
            if not links:
                raise Exception("No search results found from SerpAPI")
            
            # Fetch and summarize content for each link
            summaries = []
            for link in links:
                # Get webpage content
                page_text = await fetch_webpage_text_async(link)
                if not page_text:
                    print(f"Failed to fetch content from {link}")
                    continue
                
                # Generate summary
                summary = await summarize_text_async(text_query, page_text)
                if summary:
                    summaries.append((link, summary))
                else:
                    print(f"Failed to generate summary for {link}")
            
            if not summaries:
                raise Exception("Failed to generate any valid summaries. Please check API keys and permissions.")
            
            return summaries

        # Run the async pipeline
        summaries = asyncio.run(search_pipeline())
        
        # Format the results
        if summaries:
            tool_returned_str = "[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:\n\n"
            for i, (link, summary) in enumerate(summaries, 1):
                tool_returned_str += f"{i}. {link}\n{summary}\n\n"
            
            tool_stat = {
                "success": True,
                "num_results": len(summaries),
            }
        else:
            raise Exception("No summaries generated")

    except Exception as e:
        error_msg = str(e)
        print(f"Error in text search: {error_msg}")
        if "401" in error_msg:
            tool_returned_str = "[Text Search Results] Authentication failed. Please check API keys and permissions."
        elif "429" in error_msg:
            tool_returned_str = "[Text Search Results] Rate limit exceeded. Please try again later."
        else:
            tool_returned_str = f"[Text Search Results] Search failed: {error_msg}"
        
        tool_stat = {
            "success": False,
            "num_results": 0,
            "error": error_msg
        }

    return tool_returned_str, tool_stat
