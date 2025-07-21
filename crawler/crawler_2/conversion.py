from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import duckdb
import gzip
import re

connection = duckdb.connect("data.db")

all_docs = connection.execute("SELECT * FROM documents").fetchall()

connection.close()

connection_new = duckdb.connect("final.db")

with open("new.sql", "r") as f:
    script = f.read()

connection_new.execute(script)

#All uninteresting tags in the html, that do not contain useful text 
UNWANTED_TAGS = {"script", "style", "noscript", "template", "header", "footer", "nav", "aside", "svg", "img", "meta", "link"}

#Get useful text from html
def visible_text(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    
    image_url = extract_main_image_url(soup, base_url)
    for tag in soup(UNWANTED_TAGS):
        tag.decompose()
    
    text = soup.get_text(separator=" ", strip=True)
    return (re.sub(r"\s{2,}", " ", text), image_url)

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

amount = len(all_docs)

#Put all data into final.db
for i in range(amount):
    url = all_docs[i][1]
    title = all_docs[i][2]
    base = urlparse(url)
    base_url = str(base.scheme + "://" + base.netloc)

    html = str(gzip.decompress(all_docs[i][3]), encoding="utf-8")
    (content, image_url) = visible_text(html, base_url)

    compressed = gzip.compress(bytes(content, encoding="utf-8"))

    connection_new.execute(f"INSERT INTO documents(link, title, content, image_url) VALUES(?, ?, ?, ?)", (url, title, compressed, image_url))

    # print(i)

connection_new.close()