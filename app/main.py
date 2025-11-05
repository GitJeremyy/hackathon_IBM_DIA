from fastapi import FastAPI
from .routers import embedding, rag
import os
import logging

app = FastAPI()

app.include_router(embedding.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    return {"message": "Type /docs to access API documentation!"}