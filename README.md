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
│   ├── models/
│   ├── config.py
│   └── trainer.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   └── ...
│
├── retrieval_engine/
│   ├── bm25_retriever.py
│   ├── cross_encoder_reranker.py
│   ├── dense_retriever.py
│   ├── retrieval_engine.py
│   ├── rocchio_prf.py
│   └── rrf.py
│
├── tests/
│   ├── retrieval_engine/
│   └── ... (yet to be implemented)
│
├── conftest.py
├── endpoints.py
├── README.md
└──requirements.txt
```

## Installation

1. If not already done, clone the repository:
   ```bash
   git clone git@github.com:jonez-x/mse-project-group1.git
    cd mse-project-group1
    ```
   
2. Set up a virtual environment (optional but recommended):
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   
3. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```
   
## Usage
1. Start the FastAPI server:
   ```bash
   python endpoints.py
   ```

To make sure the server is running correctly, you can visit http://localhost:8000/docs in your web browser to see the API documentation and test the endpoints.

2. Start the frontend:
   ```bash
   cd frontend
   npm install  
   npm run dev
   ```
   
Then, to use the search engine, navigate to http://localhost:5173 in your web browser.


## Testing
To run the tests, you can use pytest and run the following command in the root directory of the project:
```bash
python -m pytest tests/retrieval_engine/ -v -s
```


