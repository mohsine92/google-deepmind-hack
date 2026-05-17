from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title = "BioLens AI",
    description = "BioLens AI is a powerful tool designed to analyze and interpret complex biological data, providing insights and predictions to advance research and discovery in the field of biology.",
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials= True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": " BioLens AI is running !"}