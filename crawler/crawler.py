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
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from newspaper import Article
import chardet
import unicodedata
from urllib.parse import urlparse, parse_qs, urlencode
import random


# visited url for dublicate check and result list
VISITED = set()     # global set of all visited urls
results = []        # result batch
DB_FILENAME = "tuebingen_crawl.duckdb"      # data base name

# Optional header for crawler introduction
HEADERS = {
    'User-Agent': 'University-Tübingen-Research-Crawler/1.0'
}

# Keywords to detect relevance
KEYWORDS = [
    "tübingen", 
    "tuebingen", 
    "university", 
    "uni", 
    "baden", 
    "württemberg", 
    "neckar", 
    "old town",
    "swabian",
    "swabia", 
    "tübinger", 
    "tübingen's",
    "tübingen university",
    "hölderlin",
    "bebenhausen",
    "hohenzollern",
]


# Download the NLTK stopwords if not already done
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

# clean the extracted text and shorten it to the excerpt
def clean_and_process_text(text, max_length=1000):
    """
    Clean and process raw text, removing unwanted elements like HTML tags, ads, dates, 
    short paragraphs, and normalizing characters.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalize the characters (remove accents, etc.)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    # Remove dates (e.g., "12/10/2023", "10-12-2023")
    text = re.sub(r'\d{1,2}[\-/]\d{1,2}[\-/]\d{4}', '', text)

    # Remove short, irrelevant paragraphs (optional, can be customized)
    paragraphs = text.split("\n")
    long_paragraphs = [p for p in paragraphs if len(p) > 50]
    text = "\n".join(long_paragraphs)

    # Tokenize and remove stopwords (optional)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Join the words back together into a cleaned excerpt
    cleaned_text = ' '.join(words)

    # Limit the text to the specified maximum length
    cleaned_text = cleaned_text[:max_length]

    return cleaned_text

# Function to remove footer and sidebar
def remove_footer_and_sidebar(soup):
    for footer in soup.find_all(['footer', 'aside']):
        footer.decompose()  # Remove these elements completely
    return soup

# Function to remove JavaScript, CSS, and noscript
def remove_unwanted_tags(soup):
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()  # Remove these tags completely
    return soup

# Function to remove ads (CSS pattern)
def remove_ads(soup):
    for ad_block in soup.find_all(class_=re.compile('ad|advertisement|banner', re.IGNORECASE)):
        ad_block.decompose()  # Remove ad blocks
    return soup

# Main function to clean the entire HTML content (if it's HTML)
def clean_html_content(soup):
    soup = remove_unwanted_tags(soup)  # Remove unwanted tags (script, style, noscript)
    soup = remove_footer_and_sidebar(soup)  # Remove footer and sidebars
    soup = remove_ads(soup)  # Remove advertisements
    return soup

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
    # Check for any of the keywords
    if any(keyword.lower() in text for keyword in KEYWORDS):
        return True
    # Check for additional specific patterns
    location_patterns = [
        r"\btübingen\b",                # "Tübingen"
        r"\b(tuebingen|university)\b",  # Combination like "Tuebingen University"
        r"\b(baden-württemberg)\b",     # Detect the region
        r"\bneckar\b",                  # Detect the "Neckar"
        r"\b(old town)\b",              # Old town reference
    ]
    # Match specific patterns
    if any(re.search(pattern, text) for pattern in location_patterns):
        return True
    # Optional: Check for the usage of location-specific landmarks or figures (e.g., Hölderlin)
    if "hölderlin" in text:
        return True
    return False

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

# Function to clean the navigation spam
def is_navigation_spam(link_text, href):
    # List of common navigation words to remove
    navigation_keywords = [
        "next", "previous", "back to top", "home", "contact", "about", "privacy", 
        "terms", "back", "forward", "pagination", "page", "search"
    ]
    # Check if the link text or href contains navigation keywords
    if any(keyword.lower() in link_text.lower() for keyword in navigation_keywords):
        return True
    # Optionally: Check if the link is a paginated URL
    if re.search(r'(\?page=\d+|\#)', href):  # Matches ?page=2 or fragment links (#)
        return True
    # Check website structure links
    if href.lower() in ["#", "javascript:void(0)"]:
        return True
    return False

# Function to clean newsletter spam
def is_newsletter_spam(link_text, href):
    # List of common newsletter subscription keywords
    newsletter_keywords = [
        "subscribe", "newsletter", "sign up", "join", "register", 
        "get updates", "receive emails", "opt-in", "mailing list", "email alerts"
    ]
    # Check if the link text or href contains newsletter-related keywords
    if any(keyword.lower() in link_text.lower() for keyword in newsletter_keywords):
        return True
    # Check if the href contains common newsletter signup or mailto links
    if re.search(r'/(subscribe|signup|newsletter|join|register)/', href.lower()):
        return True
    # Check for mailto links (common in newsletter signups)
    if href.lower().startswith('mailto:'):
        return True
    
    return False

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
    query = parse_qs(parsed.query)
    # Clean query params like utm_* or session identifiers
    cleaned_query = {key: value for key, value in query.items() if not key.startswith("utm_")}
    parsed = parsed._replace(query=urlencode(cleaned_query, doseq=True))
    return parsed.geturl()
    #return parsed._replace(fragment='', query='').geturl()

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
    sys.stdout.write(f"\r[Crawling] {url} - crawled {len(VISITED)} URLs")
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
        # Clean HTML contect (ads, unwanted tags, footer and side bar)
        cleaned_soup = clean_html_content(soup)
        text = cleaned_soup.get_text(separator=' ', strip=True)
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
        # Clean the text and shorten it to the excerpt
        excerpt = clean_and_process_text(text, max_length=1000)

        # Save metadata
        results.append({
            "url": url,
            "title": title,
            "excerpt": excerpt,
            "main_image": main_image_url,
            "favicon": fav_url
        })

        # Follow internal links
        for link in soup.find_all("a", href=True):
            abs_url = urljoin(url, link['href'])
             # Skip the navigation or newsletter spam
            link_text = link.get_text(strip=True)
            if is_navigation_spam(link_text, abs_url) or is_newsletter_spam(link_text, abs_url):
                continue
            if is_valid_url(abs_url):
                if tldextract.extract(abs_url).top_domain_under_public_suffix  == domain:
                    visited_urls = crawl(abs_url, domain, depth + 1, max_depth, results, visited_urls, max_results)

        # Delay to not fire continious requests on one site
        delay = random.uniform(1, 3)
        time.sleep(delay)

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
        "https://www.uni-tuebingen.de/en/",
        "https://www.stadtmuseum-tuebingen.de/english/",
        "https://www.germany.travel/en/cities-culture/tuebingen.html",
        "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tuebingen-germany/",
        "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tuebingen-germany/",
        "https://germanyfootsteps.com/things-to-do-in-tubingen/",
        "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
        "https://en.wikipedia.org/wiki/T%C3%BCbingen",
        "https://www.mygermanuniversity.com/cities/Tuebingen",
        "https://www.wayfaringwithwagner.com/visiting-tuebingen-in-wintertime/",
        "https://www.visit-bw.com/en/article/tubingen/df9223e2-70e5-4ee9-b3f2-cd2355ab8551#/",
        "https://www.thelocal.de/search?q=t%C3%BCbingen",
        "https://theculturetrip.com/europe/germany/articles/the-top-10-things-to-do-in-tubingen-germany",
        "https://theculturetrip.com/europe/germany/articles/how-to-spend-24-hours-in-tubingen-germany",
        "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
        "https://www.germansights.com/tubingen/",
        "https://www.britannica.com/place/Tubingen-Germany",
        "https://www.visit-bw.com/en/article/old-town-of-tubingen/c82ceb67-f78d-4b4e-8911-3972ca794cbc",
        "https://velvetescape.com/things-to-do-in-tubingen/",
        "https://justinpluslauren.com/things-to-do-in-tubingen-germany/",
        "https://globaltravelescapades.com/things-to-do-in-tubingen-germany/",
        "https://www.outdooractive.com/en/places-to-eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
        "https://www.accuweather.com/en/de/t%C3%BCbingen/72070/weather-forecast/167215",
        "https://weather.com/weather/tenday/l/T%C3%BCbingen+Baden+W%C3%BCrttemberg+Germany?canonicalCityId=7422e9d446d9837997972a38336fceb5",
        "https://www.weather-forecast.com/locations/Tubingen/forecasts/latest"
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
                depth_limit = 8
            visited_urls = crawl(seed, domain, max_depth=depth_limit)

    print(f"\n[Finished]\n")

    # Save remaining results
    if len(results) > 0:
        save_results_and_visited(results, visited_urls)
    
    # check database
    con = duckdb.connect("tuebingen_crawl.duckdb")
    print(con.execute("SELECT COUNT(*) FROM crawl_results").fetchone())
    con.close()
    
    import pandas as pd
    con = duckdb.connect("tuebingen_crawl.duckdb")
    df = con.execute("SELECT * FROM crawl_results LIMIT 5").fetchdf()
    print(df[["title", "url"]])
    con.close()
