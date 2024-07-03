# app/main.py

from fastapi import FastAPI
from app.routes import train, predict

app = FastAPI()

app.include_router(train.router)
app.include_router(predict.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
