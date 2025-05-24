from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from api.predict import predict_snake_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting Down...")


app = FastAPI(title="Snake Identification API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    # Validate file extension
    if not file.filename.lower().endswith(("jpg", "jpeg", "png")):
        raise HTTPException(
            status_code=400,
            detail="Only JPG/JPEG/PNG files are allowed"
        )

    try:
        # Reset and read file content
        await file.seek(0)
        img_bytes = await file.read()
        print(f"Bytes length: {len(img_bytes)}")

        # Validate content exists
        if not img_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Process the file
        result = await predict_snake_info(img_bytes)
        if "error" in result:
            return JSONResponse(content=result, status_code=400)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=7813, reload=True)
