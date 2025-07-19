from logger import Logger
from crawler import Crawler, signal_handler

from urllib.parse import urljoin, urlparse

import asyncio
import aiohttp
import signal
import os
import duckdb

SEED = [
    #university
    "https://uni-tuebingen.de/en/",
    
    #food
    "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",

    #weather
    "https://www.accuweather.com/en/de/t%C3%BCbingen/72070/weather-forecast/167215",
    
    #trips
    "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
    "https://wanderlog.com/list/geoCategory/284004/best-coffee-shops-and-best-cafes-in-tubingen",

    #other stuff
    "https://tuebingenresearchcampus.com/",
    "https://www.komoot.com/guide/210692/attractions-around-landkreis-tuebingen"
]

connection = duckdb.connect("data.db")

with open("setup.sql", "r") as f:
    sql_script = f.read()

connection.execute(sql_script)

async def test():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False, limit=100, limit_per_host=5)) as session:
        c = Crawler(SEED, session, connection)
        await c.start()

signal.signal(signal.SIGINT, signal_handler)
asyncio.run(test())

connection.close()