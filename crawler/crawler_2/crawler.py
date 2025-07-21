from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from simhash import Simhash

from logger import Logger
# from readability import Document

import asyncio
import aiohttp
import json
import os
import collections
import time
import csv
import datetime
import re
import duckdb
import gzip

UNWANTED_TAGS = {"script", "style", "noscript", "template", "header", "footer", "nav", "aside", "svg", "img", "meta", "link"}

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    print("Received shutdown signal.")
    shutdown_event.set()

def visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(UNWANTED_TAGS):
        tag.decompose()
    
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s{2,}", " ", text)

class Document:
    def __init__(self, link, title, content):
        self.link = str(link)
        self.title = str(title)
        self.content = str(content)

class Crawler:
    def __init__(self, seed : list, session : aiohttp.ClientSession, connection : duckdb.DuckDBPyConnection):
        self.frontier = collections.deque(seed)
        self.visited = set()

        self.base_urls : dict[str, RobotFileParser] = {}

        self.session = session
        self.dbconnection = connection
        
        self.langs = ["en", "en-us", "eng", "english", "englisch"]
        self.keywords = ["tübingen", "tuebingen", "tubingen"]

        self.info_loc = "crawl_info/"
        self.robot_loc = "/robots.txt"
        self.state_loc = self.info_loc + "state.json"

        self.MAX_CON_TASKS = 10

        self.delay = 1
        self.page = 0
        self.docs : list[Document] = []

        self.logger = Logger()

        self.time_out = aiohttp.ClientTimeout(total=15, connect=5, sock_connect=5, sock_read=5)

        self.sema_base_urls = asyncio.Semaphore()
        self.sema_frontier = asyncio.Semaphore()

        self.simhashes = []
        self.simhashe_threshold = 3     # adjust for granularity

    def load_state(self):
        if os.path.exists(self.state_loc):
            with open(self.state_loc, "r") as f:
                data = json.load(f)
                self.visited = set(data['visited'])
                self.frontier = collections.deque(data['logged'])
        else:
            with open(self.state_loc, "w") as f:
                pass


    def save_state(self):
        with open(self.state_loc, "w") as f:
            json.dump({
                "visited" : [url for url in self.visited],
                "logged" : [url for url in self.frontier]
            }, f)

    async def start(self):
        if not os.path.exists(self.info_loc):
            os.mkdir(self.info_loc)
        await self.logger.print_msg("Starting crawler.", "i")
        start_crawl_time = time.time()

        self.logger.create_log_file()
        self.load_state()

        start_docs = len(self.visited)

        tasks = []

        print(len(self.frontier))
        while not shutdown_event.is_set():
            while len(tasks) < self.MAX_CON_TASKS and len(self.frontier) > 0:
                task = asyncio.create_task(self.wrapped_crawl(self.frontier.pop()))
                tasks.append(task)

            finished, tasks = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

            tasks = [ta for ta in tasks]

            for fini in finished:
                try:
                    await fini
                except Exception as e:
                    await self.logger.print_msg(f"Problem with finished, EXCEPTION: {e}", "e")

            if shutdown_event.is_set():
                break

            if len(self.frontier) == 0:
                shutdown_event.set()
                break

        if shutdown_event.is_set():
            self.save_state()

            await self.logger.print_msg(f"Closing crawler.\n", "i")

            end_crawl_time = time.time()
            end_docs = len(self.visited)

            await self.logger.print_msg(f"TOTAL CRAWL TIME: {datetime.timedelta(seconds=end_crawl_time - start_crawl_time)}", "i")
            await self.logger.print_msg(f"TOTAL VISITED PAGES: {end_docs - start_docs}", "i")

    async def get_robot(self, base_url):
        robots_loc = urljoin(base_url, self.robot_loc)
        try:
            async with self.session.get(robots_loc) as response:
                if response.status == 200:
                    robots_txt = await response.text()
                else:
                    await self.logger.print_msg(f"STATUS: {response.status}, Could not get robots.txt.", "e")
                    return None
        except Exception as e:
            await self.logger.print_msg(f"Could not fetch robots.txt at {robots_loc}, EXCEPTION: {e}", "e")

        robot = RobotFileParser()
        robot.set_url(robots_loc)
        robot.read()
        robot.parse(robots_txt.splitlines())

        return robot
    
    # Function to compare simhash against allready seen simhashes
    async def compare_simhashes(self, simhash):
        # Check by thresholding
        for seen_simh in self.simhashes:
            if simhash.distance(seen_simh) <= self.simhashe_threshold:
                await self.logger.print_msg("DUBLICATE CONTEND", "i")
                return True
        self.simhashes.append(simhash)
        return False

    async def parse(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        lang = soup.find("html").get("lang")
        title = soup.find("title").get_text()
        
        if not lang in self.langs:
            return
        links = []

        content = html #visible_text(html)
        await self.logger.print_msg(f"LANG: {lang}, TITLE: {title}", "i")

        # Check simhash over document text against seen simhashes
        simh = Simhash(content)
        if await self.compare_simhashes(simh):
            return

        for link_tag in soup.find_all("a"):
            link = urljoin(url, link_tag.get("href"))
            links.append(link)

        return (lang, title, links, content)

    async def find_keyword(self, url):
        for s in self.keywords:
            if str(url).find(s) != -1:
                return True

    async def crawl(self, url):
        if url in self.visited:
            return
        
        if not await self.find_keyword(url):
            return

        try: 
            async with self.session.get(url, timeout=self.time_out) as response:
                if response.status != 200:
                    await self.logger.print_msg(f"STATUS: {response.status}, Could not fetch {url}", "e")
                    return
                
                raw = await response.read()

                #get the encoding type
                enc = response.charset
                if enc is None:
                    head = raw[:4096].decode("ascii", "ignore")
                    m = re.search(r'charset=["\']?([\w-]+)', head, re.I)
                    if m:
                        enc = m.group(1)
                enc = enc.lower() or "utf-8"
                html = raw.decode(enc, "replace")

                await self.logger.print_msg(f"Encoder: {enc}", "i")

                base = urlparse(url)
                base_url = str(base.scheme + "://" + base.netloc)

                if not base_url in self.base_urls:
                    #check robot
                    robot = await self.get_robot(base_url)

                    async with self.sema_base_urls:
                        self.base_urls[base_url] = robot
                
                if not self.base_urls[base_url] is None:
                    delay = self.base_urls[base_url].crawl_delay("*")
                    if not self.base_urls[base_url].can_fetch("*", url):
                        await self.logger.print_msg(f"Could not fetch {url} because robots.txt.", "w")
                        return

                links = await self.parse(url, html)

                if links:
                    title = links[1]
                    content = gzip.compress(bytes(links[3], encoding="utf-8"))
                    self.dbconnection.execute(f"INSERT INTO documents(link, title, content) VALUES(?, ?, ?)", (url, title, content))

                self.frontier = collections.deque(set.union(set(self.frontier), set(links[2])))
                
                self.visited.add(url)

                await asyncio.sleep(0 if delay is None else delay)
                return
            
        except Exception as e:
            await self.logger.print_msg(f"Could not fetch {url}, EXCEPTION: {e}", "e")
            await asyncio.sleep(self.delay)


    async def wrapped_crawl(self, url):
        async with self.sema_frontier:
            await self.crawl(url)