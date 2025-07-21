# mse-project-group1

## Group members:

- Marco Bäuerle
- Sebastian Dubiel
- Nicolas Schmitt
- Jonas Taigel
- Karam Preet Singh Wolfrum

## Project Description

This project implements a modern search engine tailored to the city of Tübingen.
It combines custom web crawling, multiple retrieval strategies (BM25, dense retrieval, re-ranking),
and an interactive frontend featuring both list and Tinder-like views with query heatmaps.

## Project Structure

```
mse-project-group1/
├── autocomplete_system/
│   ├── data/
│   │   └── loader.py               # Load the duckdb database for ngram training
│   ├── models/
│   │   ├── base.py                 # Abstract class for autocomplete models
│   │   ├── datamuse.py             # Datamuse API integration for autocomplete
│   │   └── ngram.py                # N-gram model implementation
│   ├── services/
│   │   └── autocomplete.py         # Autocomplete service for handling requests
│   └── trainer.py                  # Scipt to train the autocomplete models (only for ngram)
│
├── batch/                          # Batch processing scripts for the search engine
│   ├── queries/queries.txt         # Queries for batch processing 
│   └── run_queries.py              # Script to run batch queries 
│
├── crawler/                        # Web crawlers for collecting data
│   ├── crawler_2/
│   ├── crawler.py
│   └── tuebingen_crawl.duckdb      # DuckDB database for storing crawled data 
│
├── frontend/                       # React frontend for the search engine
│   ├── public/
│   ├── src/
│   └── ...
│
├── retrieval_engine/
│   ├── core/
│   │   └── engine.py               # Main RetrievalEngine class
│   ├── docs/
│   │   └── document_store.py       # Document and DocumentStore classes for managing documents
│   ├── enhancement/
│   │   ├── reranking.py            # CrossEncoderReRanker for reranking results
│   │   └── rocchio_prf.py          # Rocchio Pseudo Relevance Feedback
│   ├── fusion/
│   │   └── rrf.py                  # Reciprocal Rank Fusion
│   └── retrievers/
│       ├── dense.py                # Dense semantic retrieval 
│       └── sparse.py               # BM25 sparse retrieval
│
├── tests/                          # Unit tests for the retrieval engine
│   ├── autocomplete_system/
│   ├── retrieval_engine/
│   └── conftest.py
│
├── config.py                       # Configuration file for the backend
├── endpoints.py                    # FastAPI endpoints for the search engine
├── README.md
├── requirements.txt
└── start.sh                        # Script to start frontend + backend
```

## Installation

1. If not already done, clone the repository:

```bash
git clone git@github.com:jonez-x/mse-project-group1.git
 cd mse-project-group1
 ```

Alternatively, use HTTPS to clone the repository:

```bash
 git clone https://github.com/jonez-x/mse-project-group1.git
 cd mse-project-group1
 ```

2. Set up a virtual environment (optional but recommended):

```bash
python3.13 -m venv .venv
source .venv/bin/activate 
```

On windows, you can use the following command to activate the virtual environment:

```bash
python3.13 -m venv .venv
.\.venv\Scripts\activate
```

This project was developed and tested with Python 3.13, so make sure to use this version or a compatible one.

3. Install the required packages:

```bash
 pip install -r requirements.txt
 ```

## Usage

**Important** : The database (final.zip) is zipped to reduce the size, it needs to be unzipped to start the search
engine.

1. Start the FastAPI server:

```bash
python endpoints.py
```

For the autcompletion system, we use an API-based model as default, which is the Datamuse API.
If you want to use the ngram model, simply set the model flag when starting the server:

```bash
python endpoints.py --model="ngram"
```

To make sure the server is running correctly, you can visit http://localhost:8000/docs in your web browser to see the
API documentation and test the endpoints.

2. Start the frontend:

```bash
cd frontend
npm install  
npm run dev
```

Then, to use the search engine, navigate to http://localhost:5173 in your web browser.

Alternatively, you can use the script `start.sh` to start the frontend and backend automatically:

```bash
./start.sh
```

## Testing

To run the tests, you can use pytest and run the following command in the root directory of the project:

```bash
python -m pytest tests/retrieval_engine/ -v -s
```

## Web Crawling

To start the Web crawler simply execute the main.py in the crawler folder:

 ```bash
python crawler/crawler_2/main.py
```

The crawler will run, until there is a shutdown signal CTRL+C or the frontier is empty.
After the crawler is finished there will be a crawl_info/ folder, which contains the visited and logged URLs in a JSON
file and a log file.
Additionally there will be a newly created .db file (data.db or another similar name), which contains the entire data of
the crawling process.

The data.db file consists of an ID, URL, Title and the compressed HTML.
The rest of the code will need another database structure. Mainly ID, URL, Title, compressed content of the HTML,
image_url (if found at all).

Therefore, a conversion is needed, which extracts the visible text and the main image from the HTML file. This will
create a file called final.db.
To start the conversion script, simply run:

 ```bash
python crawler/crawler_2/conversion.py
```
