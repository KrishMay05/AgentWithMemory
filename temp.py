#!/usr/bin/env python3
"""
test_search_web.py

Standalone test harness for the `search_web` function.
Includes the full implementation of `search_web` so you can verify your Google CSE keys.

Usage:
  export GOOGLE_API_KEY=your_key
  export GOOGLE_CSE_ID=your_cse_id
  pip install google-api-python-client wikipedia bs4 requests
  python test_search_web.py
"""
import os
import requests
from bs4 import BeautifulSoup
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def search_web(query: str, sentences: int = 3) -> str:
    """
    Perform a quick lookup via Wikipedia and return the first few sentences.
    Falls back to a Google-Custom-Search snippet fetch (up to 3 results), 
    and if a snippet isn’t available, scrapes the first meaningful <p> from the page.
    Gracefully handles expired/invalid API keys and HTTP errors.
    """
    # 1) Try Wikipedia first
    # wikipedia.set_lang("en")
    # try:
    #     return wikipedia.summary(query, sentences=sentences, auto_suggest=False, redirect=True)
    # except DisambiguationError as e:
    #     choice = e.options[0]
    #     try:
    #         return wikipedia.summary(choice, sentences=sentences)
    #     except Exception:
    #         pass
    # except PageError:
    #     pass

    # 2) Google CSE fallback
    api_key = "AIzaSyDoKdONIpBj0gUulxB45ACm4kN2BLlmBZM"
    cse_id  = "f04f2981f50a045e9"
    if not api_key or not cse_id:
        return "No Google API key or CSE ID configured."

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res     = service.cse().list(q=query, cx=cse_id, num=3).execute()
    except HttpError as e:
        # Handle HTTP errors from the Google API (e.g., expired/invalid API key)
        try:
            error_json = e.error_details or str(e)
        except AttributeError:
            error_json = str(e)
        return f"Google API error: {error_json}"
    except Exception as e:
        return f"Search failed: {e}"

    items = res.get("items", [])
    if not items:
        return "No results found."

    snippets = []
    for item in items:
        snippet = item.get("snippet")
        if snippet:
            snippets.append(snippet)
            continue
        link = item.get("link")
        try:
            resp = requests.get(link, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            for p in soup.find_all("p"):
                text = p.get_text().strip()
                if len(text) > 50:
                    snippets.append(text)
                    break
        except Exception:
            continue

    return "\n\n".join(snippets) if snippets else "Found a page, but couldn’t extract a summary."


def main():
    # Echo environment variables
    print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
    print("GOOGLE_CSE_ID: ", os.getenv("GOOGLE_CSE_ID"))

    # Sample queries
    tests = [
        "Barack Obama current age",
        "Purdue University founding date",
        "Newest Sidemnen video",
    ]

    for query in tests:
        print(f"\n=== Query: {query} ===")
        try:
            result = search_web(query)
        except Exception as e:
            result = f"Error: {e}"
        print(result)


if __name__ == "__main__":
    main()
