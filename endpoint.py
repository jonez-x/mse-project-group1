from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    print(f"Received query: {q}")
    
    if q == "result1":
        return {"query": q, "results": ["result1"]}
    elif q == "result2":
        return {"query": q, "results": ["result2"]}
    elif q == "result3":
        return {"query": q, "results": ["result3"]}

    return {"query": q, "results": "No results found"}