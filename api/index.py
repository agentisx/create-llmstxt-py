import os
import json
import logging
import requests
from openai import OpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class FirecrawlLLMsTextGenerator:
    """Generate llms.txt files using Firecrawl and OpenAI."""
    
    def __init__(self, firecrawl_api_key: str, openai_api_key: str):
        """Initialize the generator with API keys."""
        self.firecrawl_api_key = firecrawl_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.firecrawl_base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
    
    def map_website(self, url: str, limit: int = 100) -> List[str]:
        """Map a website to get all URLs."""
        logger.info(f"Mapping website: {url} (limit: {limit})")
        
        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/map",
                headers=self.headers,
                json={"url": url, "limit": limit, "includeSubdomains": False, "ignoreSitemap": False}
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("success") and data.get("links"):
                urls = data["links"]
                logger.info(f"Found {len(urls)} URLs")
                return urls
            else:
                logger.error(f"Failed to map website: {data}")
                return []
        except Exception as e:
            logger.error(f"Error mapping website: {e}")
            return []
    
    def scrape_url(self, url: str) -> Optional[Dict]:
        """Scrape a single URL."""
        logger.debug(f"Scraping URL: {url}")
        
        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/scrape",
                headers=self.headers,
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True, "timeout": 30000}
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("success") and data.get("data"):
                return {
                    "url": url,
                    "markdown": data["data"].get("markdown", ""),
                    "metadata": data["data"].get("metadata", {})
                }
            else:
                logger.error(f"Failed to scrape {url}: {data}")
                return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def generate_description(self, url: str, markdown: str) -> Tuple[str, str]:
        """Generate title and description using OpenAI."""
        logger.debug(f"Generating description for: {url}")
        
        prompt = f"""Generate a 9-10 word description and a 3-4 word title of the entire page based on ALL the content one will find on the page for this url: {url}. This will help in a user finding the page for its intended purpose.

Return the response in JSON format:
{{
    "title": "3-4 word title",
    "description": "9-10 word description"
}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise titles and descriptions for web pages."},
                    {"role": "user", "content": f"{prompt}\n\nPage content:\n{markdown[:4000]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=100
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("title", "Page"), result.get("description", "No description available")
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Page", "No description available"
    
    def process_url(self, url: str) -> Optional[Dict]:
        """Process a single URL: scrape and generate description."""
        scraped_data = self.scrape_url(url)
        if not scraped_data or not scraped_data.get("markdown"):
            return None
        
        title, description = self.generate_description(
            url, 
            scraped_data["markdown"]
        )
        
        return {
            "url": url,
            "title": title,
            "description": description,
            "markdown": scraped_data["markdown"],
        }
    
    def generate_llmstxt_data(self, url: str, max_urls: int = 100) -> Tuple[List[Dict], List[Dict]]:
        """Generate llms data for a website."""
        logger.info(f"Generating llms data for {url}")
        
        # Step 1: Map the website
        urls = self.map_website(url, max_urls)
        if not urls:
            raise ValueError("No URLs found for the website")
        
        urls = urls[:max_urls]
        
        all_results = []
        batch_size = 10
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
                
                futures = {executor.submit(self.process_url, u): u for u in batch}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        url_failed = futures[future]
                        logger.error(f"Failed to process {url_failed}: {e}")
        
        # Format the output into the desired lists
        llms_data = [{"url": res["url"], "title": res["title"], "description": res["description"]} for res in all_results]
        llms_full_data = [{"url": res["url"], "title": res["title"], "description": res["description"], "markdown": res["markdown"]} for res in all_results]
        
        return llms_data, llms_full_data

@app.route('/api/generate-llms', methods=['POST'])
def generate_llms_endpoint():
    """
    HTTP endpoint to trigger the LLM text generation.
    Expects a JSON payload with a 'url' key.
    """
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    
    url = data['url']
    max_urls = data.get('max_urls', 100)

    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not firecrawl_api_key or not openai_api_key:
        return jsonify({"error": "API keys not configured"}), 500

    try:
        generator = FirecrawlLLMsTextGenerator(firecrawl_api_key, openai_api_key)
        llms_data, llms_full_data = generator.generate_llmstxt_data(url, max_urls)
        
        response_data = {
            "llms": llms_data,
            "llms_full": llms_full_data
        }
        
        return jsonify(response_data), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# Vercel needs an entry point, which Flask provides.
# Vercel will look for this `app` variable.

