from fastapi import FastAPI
from app.api import routes
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="FaceAI Cloud", version="0.1")

# include API router
app.include_router(routes.router)

# static sitemap & robots
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def on_startup():
    # create downloads or model folders if needed
    os.makedirs("models", exist_ok=True)

@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
