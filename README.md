# mse-project-group1

## Group members:

- Marco Bäuerle
- Sebastian Dubiel
- Nicolas Schmitt
- Jonas Taigel
- Karam Preet Singh Wolfrum

## Project Description

Yet to be written here...

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
│   ├── retrieval_engine/
│   └── ... (yet to be implemented)
│
├── config.py                       # Configuration file for the retrieval engine
├── conftest.py
├── endpoints.py
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
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

This project was developed and tested with Python 3.13, so make sure to use this version or a compatible one.

3. Install the required packages:

```bash
 pip install -r requirements.txt
 ```

## Usage

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

Alernatively, you can use the script `start.sh` to start the frontend automatically:

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
After the crawler is finished there will be a crawl_info/ folder, which contains the visited and logged URLs in a JSON file and a log file.

After the crawler is done, a conversion is needed, which extracts the visible text and the main image from the HTML file.
To start the conversion script, simply run:

 ```bash
python crawler/crawler_2/conversion.py
```
