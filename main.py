from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model("anomaly_model.h5")

app = FastAPI(title="Image Anomaly Detection API")

# CORS setup for frontend (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Shape for model
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_img = preprocess_image(contents)

        reconstructed = model.predict(input_img)
        loss = np.mean((input_img - reconstructed) ** 2)

        threshold = 0.01  # You can tune this from earlier evaluation
        status = "Anomalous" if loss > threshold else "Normal"

        return JSONResponse(content={
            "prediction": status,
            "reconstruction_error": float(loss)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)