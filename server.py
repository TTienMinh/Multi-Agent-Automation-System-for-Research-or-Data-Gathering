from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the arXiv Abstract Fetcher API!"}

@app.get("/print_abstracts/")
def print_abstracts():
    with open("arxiv_abstracts.txt", "r", encoding="utf-8") as f:
        abstracts = f.read()
    return {"abstracts": abstracts}