import os, time, math, re, requests
from datetime import datetime, timezone
from urllib.parse import urlencode
from bs4 import BeautifulSoup

# Optional: pip install trafilatura
import trafilatura
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"}

def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["age", "born", "birthdate", "founding date", "founded", "founded date"]):
        return "entity_fact"
    if any(k in ql for k in ["latest", "newest", "today", "this week", "just released", "video"]):
        return "fresh"
    if any(k in ql for k in ["what is", "define", "meaning of"]):
        return "definition"
    return "general"

# --- Specialized handlers ---
def try_wikipedia_api(q: str, sentences=3):
    # Simple, robust pull via REST summary
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(q)
        r = requests.get(url, timeout=6, headers=UA)
        if r.status_code == 200:
            js = r.json()
            txt = js.get("extract") or ""
            return txt.split(". ")[:sentences]
    except Exception:
        pass
    return None

def try_wikidata_birthdate(name: str):
    # Super-light heuristic: query Wikidata search API to get QID, then DOB
    try:
        search = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={"action":"wbsearchentities","language":"en","format":"json","search":name},
            timeout=6, headers=UA
        ).json()
        if not search.get("search"): return None
        qid = search["search"][0]["id"]
        ent = requests.get(
            "https://www.wikidata.org/wiki/Special:EntityData/{}.json".format(qid),
            timeout=6, headers=UA
        ).json()
        claims = ent["entities"][qid]["claims"]
        if "P569" in claims:
            dob = claims["P569"][0]["mainsnak"]["datavalue"]["value"]["time"]  # like '+1961-08-04T00:00:00Z'
            return dob[1:11]  # '1961-08-04'
    except Exception:
        return None

def compute_age(dob_iso: str) -> int:
    y,m,d = map(int, dob_iso.split("-"))
    today = datetime.now(timezone.utc).date()
    age = today.year - y - ((today.month, today.day) < (m, d))
    return age

# --- Google CSE with pagination ---
def google_cse(query: str, num=10, pages=2, dateRestrict=None):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")
    results = []
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        start = 1
        for _ in range(pages):
            kwargs = {"q": query, "cx": GOOGLE_CSE_ID, "num": num, "start": start, "hl":"en", "gl":"us"}
            if dateRestrict: kwargs["dateRestrict"] = dateRestrict
            res = service.cse().list(**kwargs).execute()
            items = res.get("items", [])
            results.extend(items)
            if "nextPage" not in res.get("queries", {}): break
            start += num
            time.sleep(0.2)
    except HttpError as e:
        return [], f"Google API error: {getattr(e, 'error_details', str(e))}"
    except Exception as e:
        return [], f"Search failed: {e}"
    return results, None

# --- Extraction ---
def extract_text(url: str) -> str:
    try:
        r = requests.get(url, headers=UA, timeout=8)
        r.raise_for_status()
        # Try trafilatura first
        txt = trafilatura.extract(r.text, url=url, include_comments=False, include_tables=False)
        if txt and len(txt) > 300:
            return txt
        # Fallback: basic <p> scrape
        soup = BeautifulSoup(r.text, "html.parser")
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        long_ps = [p for p in ps if len(p) > 80]
        return "\n".join(long_ps[:6])
    except Exception:
        return ""

# --- Passage ranking (very light) ---
def score_passage(q: str, text: str) -> float:
    q_terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2]
    t = text.lower()
    score = sum(t.count(term) for term in q_terms)
    score += 0.3 * sum(1 for term in q_terms if term in t.split())  # presence bonus
    return score

def top_passages(q: str, docs: list[str], k=6):
    passages = []
    for doc in docs:
        for para in doc.split("\n"):
            para = para.strip()
            if len(para) < 60: continue
            passages.append((score_passage(q, para), para))
    passages.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in passages[:k]]

# --- Main orchestrator ---
def smart_search(query: str) -> dict:
    intent = detect_intent(query)

    # 1) Specialized fast paths
    if intent == "entity_fact":
        # Try Wikipedia summary
        wiki = try_wikipedia_api(query, sentences=3)
        if wiki:
            return {"answer": " ".join(wiki), "citations": ["https://en.wikipedia.org/wiki/{}".format(query.replace(" ", "_"))]}
        # Try “age” computation via Wikidata
        if "age" in query.lower():
            name = re.sub(r"\b(current|age|years old)\b", "", query, flags=re.I).strip()
            dob = try_wikidata_birthdate(name or query)
            if dob:
                age = compute_age(dob)
                return {"answer": f"{name or query} is {age} years old (born {dob}).", "citations": ["https://www.wikidata.org/"]}

    # 2) Google as recall (+ freshness when needed)
    dateRestrict = "d7" if intent == "fresh" else None
    items, err = google_cse(query, num=10, pages=2, dateRestrict=dateRestrict)
    if err:
        return {"answer": err, "citations": []}

    urls = []
    for it in items:
        link = it.get("link")
        if link and link not in urls:
            urls.append(link)
    # Extract & rank
    texts = []
    for u in urls[:12]:  # cap extraction work
        txt = extract_text(u)
        if txt:
            texts.append(txt)

    if not texts:
        # fallback to Google snippets
        snippets = [it.get("snippet","") for it in items if it.get("snippet")]
        answer = "\n\n".join(snippets[:5]) if snippets else "No useful text extracted."
        return {"answer": answer, "citations": urls[:3]}

    passages = top_passages(query, texts, k=6)
    # Simple synthesis: join top passages (or feed to your LLM for cleaner summary)
    synthesis = " ".join(passages[:3])
    return {"answer": synthesis if synthesis else "No high-relevance passages found.", "citations": urls[:3]}


# if __name__ == "__main__":
#     queries = [
#         "Barack Obama current age",
#         "Purdue University founding date",
#         "Newest Sidemen video",
#         "What is quantum computing"
#     ]
#     for q in queries:
#         print(f"\n=== {q} ===")
#         out = smart_search(q)
#         print("Answer:", out["answer"])
#         print("Citations:", out["citations"])