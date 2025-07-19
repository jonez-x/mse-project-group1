#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from langdetect import detect
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import tldextract
import os
import sys
import duckdb

# visited url for dublicate check and result list
VISITED = set()     # global set of all visited urls
results = []        # result batch
DB_FILENAME = "tuebingen_crawl.duckdb"      # data base name

# Optional header for crawler introduction
HEADERS = {
    'User-Agent': 'University-Tübingen-Research-Crawler/1.0'
}

# Keywords to detect relevance
KEYWORDS = ["tübingen", 
            "tuebingen", 
            "university",
            "uni", 
            "baden",
            "württemberg",
            "neckar",
            "old town"]

# Check for english conntend
def is_english(text):
    # check for unrelevant short texts
    if len(text) < 200:
        return False
    try:
        return detect(text) == "en"
    except:
        return False

# Check for tübingen related conntend
def mentions_tuebingen(text):
    text = text.lower()
    return any(keyword in text for keyword in KEYWORDS)

# Only http or https sites
def is_valid_url(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme in {"http", "https"}

# Check if allowed to crawl the url based on the robots.txt file
robots_parsers = {}     # cache robot rules per domain
def can_fetch(url):
    # Create the base url
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    # Caching
    if base_url not in robots_parsers.keys():
        rp = RobotFileParser()      # use predefined class for robots file parsing
        rp.set_url(urljoin(base_url, "/robots.txt"))
        try:
            rp.read()
        except Exception as e:
            print(f"\n[Warning] Could not read robots.txt for {base_url}: {e}")
            rp = None       # if the robots.txt is unreadable, assume allowed
        robots_parsers[base_url] = rp

    rp = robots_parsers.get(base_url)
    if rp:
        # Check if our crawler is allowed for the url
        return rp.can_fetch(HEADERS['User-Agent'], url)
    return True  # if there's no robots.txt, assume allowed

# Get the favicon url
def extract_favicon_url(soup, base_url):
    # 1. Try to find <link rel="icon" ...> or similar
    icon_link = soup.find("link", rel=lambda x: x and 'icon' in x.lower())
    if icon_link and icon_link.get("href"):
        return urljoin(base_url, icon_link["href"])
    
    # 2. Fallback to default /favicon.ico
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"

# Normalize url to avoid dublicated links with different parameters
def normalize_url(url):
    parsed = urlparse(url)
    return parsed._replace(fragment='', query='').geturl()

# Try to get the most influencing image url for each site
def extract_main_image_url(soup, base_url):
    # Check for Open Graph image as most influencing image
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        return urljoin(base_url, og_image["content"])

    # Fallback: try to find the largest visible image on the site
    max_area = 0
    best_img_url = None
    for img in soup.find_all("img", src=True):
        width = int(img.get("width", 0)) if img.get("width", "").isdigit() else 0
        height = int(img.get("height", 0)) if img.get("height", "").isdigit() else 0
        area = width * height
        if area > max_area:
            max_area = area
            best_img_url = urljoin(base_url, img["src"])
    
    return best_img_url

# Create the data base if not there
def initialize_db():
    # Ensure the directory exists
    ensure_directory_exists(DB_FILENAME)
    try:
        with duckdb.connect(DB_FILENAME) as con:
            # Table for storing visited URLs
            con.execute("""
            CREATE TABLE IF NOT EXISTS visited_urls (
                url TEXT PRIMARY KEY
            );
            """)
            # Table for storing crawled seeds
            con.execute("""
            CREATE TABLE IF NOT EXISTS crawled_seeds (
                seed_url TEXT PRIMARY KEY,
                crawled BOOLEAN
            );
            """)
            # Table for storing crawl results
            con.execute("""
            CREATE TABLE IF NOT EXISTS crawl_results (
                url TEXT PRIMARY KEY,
                title TEXT,
                excerpt TEXT,
                main_image TEXT,
                favicon TEXT
            );
            """)
            print(f"[Info] Tables created (if not already present).")
    except Exception as e:
        print(f"[Error] Database initialization failed: {e}")

# Ensure the directory exists
def ensure_directory_exists(db_filename):
    db_dir = os.path.dirname(db_filename)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"[Info] Directory {db_dir} created.")

# Load progress (visited URLs and finished seeds from data base) in case of interuption
def load_progress():
    con = duckdb.connect(DB_FILENAME)
    # Load visited URLs
    visited_urls = set(row[0] for row in con.execute("SELECT url FROM visited_urls").fetchall())
    # Load finished seeds
    finished_seeds = set(row[0] for row in con.execute("SELECT seed_url FROM crawled_seeds WHERE crawled = TRUE").fetchall())
    con.close()
    return visited_urls, finished_seeds

# Mark a seed as finished
def mark_seed_finished(seed_url):
    con = duckdb.connect(DB_FILENAME)
    con.execute("INSERT OR REPLACE INTO crawled_seeds (seed_url, crawled) VALUES (?, TRUE)", (seed_url,))
    con.close()

# Function to save results  and the visited set to data base
def save_results_and_visited(results, visited_urls):
    con = duckdb.connect(DB_FILENAME)

    # Save results if they are not already in the database
    for result in results:
        url = result["url"]
        existing_url = con.execute("SELECT 1 FROM crawl_results WHERE url = ?", (url,)).fetchone()
        # Insert only if the URL doesn't exist
        if not existing_url:
            con.execute("""
            INSERT INTO crawl_results (url, title, excerpt, main_image, favicon)
            VALUES (?, ?, ?, ?, ?)
            """, (result["url"], result["title"], result["excerpt"], result["main_image"], result["favicon"]))
    # Save visited URLs
    for url in visited_urls:
        existing_visited = con.execute("SELECT 1 FROM visited_urls WHERE url = ?", (url,)).fetchone()
        # Insert only if the URL hasn't been visited
        if not existing_visited:
            con.execute("INSERT INTO visited_urls (url) VALUES (?)", (url,))

    # Create an index on the URL column of the crawl_results for faster lookups
    con.execute("CREATE INDEX IF NOT EXISTS idx_url ON crawl_results(url);")        # not necessary as primary key defines index

    con.close()
    #print(f"[Saved] {len(results)} results and {len(visited_urls)} visited URLs to {db_filename}")


# Crawler function
def crawl(url, domain, depth=0, max_depth=3, results=[], visited_urls=set(), max_results=50):
    # Save results if the reach a threshold
    if len(results) >= max_results:
        print(f"\n[Saving] Saving {len(results)} results after reaching {max_results} results.")
        save_results_and_visited(results, visited_urls)
        results.clear()         # Reset results after saving
        visited_urls.clear()    # Reset visited URLs after saving

    ## Check for visited sites (no dublicate crawling) and depth of crawling (how often to follow internal links)
    norm_url = normalize_url(url)
    if norm_url in VISITED or depth > max_depth:
        return visited_urls
    ## Check for robots.txt rules
    if not can_fetch(url):
        print(f"\n[Blocked by robots.txt] {url}")
        return visited_urls
    
    # Keep track of visited sites
    VISITED.add(norm_url)
    visited_urls.add(norm_url)      # Just for saving
    sys.stdout.write(f"\r[Crawling] {url}")
    sys.stdout.flush()
    #print(f"[Crawling] {url}")

    # Crawling
    try:
        # Get request and check for status code and html type
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
            return visited_urls
        # Define html-parser and extratct text of document
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)
        ## Check for english related conntend
        if not is_english(text):
            return visited_urls
        ## Check for Tübingen related conntend 
        if not mentions_tuebingen(text):
            return visited_urls
        # Extract document title
        title = soup.title.string.strip() if soup.title else "No Title"
        # Extract favicon url
        fav_url = extract_favicon_url(soup, url)
        # Extract most influencing image URL
        main_image_url = extract_main_image_url(soup, url)

        # Save metadata
        results.append({
            "url": url,
            "title": title,
            "excerpt": text[:300],
            "main_image": main_image_url,
            "favicon": fav_url
        })

        # Follow internal links
        for link in soup.find_all("a", href=True):
            abs_url = urljoin(url, link['href'])
            if is_valid_url(abs_url):
                if tldextract.extract(abs_url).top_domain_under_public_suffix  == domain:
                    visited_urls = crawl(abs_url, domain, depth + 1, max_depth, results, visited_urls, max_results)

        # Delay to not fire continious requests on one site
        time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"\n[Network Error] {url}: {e}")
    except Exception as e:
        print(f"\n[General Error] {url}: {e}")

    # return to attacht the rest at the end to the results
    return visited_urls


if __name__ == "__main__":
    initialize_db()  # Set up the database tables

    # Load progress from database
    VISITED, finished_seeds = load_progress()

    # Seed URLs
    seeds = [
        "https://www.tuebingen.de/en/",
        "https://www.uni-tuebingen.de/en/",
        "https://tuebingen-info.de/en/",
        "https://www.stadtmuseum-tuebingen.de/english/",
        "https://www.germany.travel/en/cities-culture/tuebingen.html",
        "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tuebingen-germany/",
        "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tuebingen-germany/",
        "https://germanyfootsteps.com/things-to-do-in-tubingen/",
        "https://en.wikipedia.org/wiki/T%C3%BCbingen",
        "https://www.mygermanuniversity.com/cities/Tuebingen",
        "https://www.wayfaringwithwagner.com/visiting-tuebingen-in-wintertime/",
        "https://www.visit-bw.com/en/article/tubingen/df9223e2-70e5-4ee9-b3f2-cd2355ab8551#/",
        "https://www.thelocal.de/search?q=t%C3%BCbingen",
        "https://theculturetrip.com/europe/germany/articles/the-top-10-things-to-do-in-tubingen-germany",
        "https://theculturetrip.com/europe/germany/articles/how-to-spend-24-hours-in-tubingen-germany",
        "https://www.germansights.com/tubingen/",
        "https://www.britannica.com/place/Tubingen-Germany"
    ]

    # Traverse the seed urls not finished to collect/extend the corpus
    visited_urls = set()
    for seed in seeds:
        if seed not in finished_seeds:
            domain = tldextract.extract(seed).top_domain_under_public_suffix 
            print(f"\n[Starting Crawl] {seed}")
            # Set custom depth for Wikipedia as too mutch wikipedia links
            if "wikipedia.org" in seed:
                depth_limit = 1
            else:
                depth_limit = 5
            visited_urls = crawl(seed, domain, max_depth=depth_limit)

    print(f"\n[Finished] Collected {len(results)} English pages mentioning Tübingen.\n")

    # Save remaining results
    if len(results) > 0:
        save_results_and_visited(results, visited_urls)
    
    # check database
    """con = duckdb.connect("tuebingen_crawl.duckdb")
    print(con.execute("SELECT COUNT(*) FROM crawl_results").fetchone())
    con.close()
    
    import pandas as pd
    con = duckdb.connect("tuebingen_crawl.duckdb")
    df = con.execute("SELECT * FROM crawl_results LIMIT 5").fetchdf()
    print(df[["title", "url"]])
    con.close()"""
