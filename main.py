from fastapi import FastAPI
from src.api.v1.routes import query_route
from src.api.v1.routes.upload import router as upload_router
app=FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Hello World"}

@app.get("/health")
def health_check():
    return {
        "status":"ok"
    }

app.include_router(query_route.router,prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")