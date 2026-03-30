from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import os
from autodml import Autodml
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AutoDML API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Welcome to AutoDML API "}


@app.post("/train")
async def train_model(target: str, file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)

        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file format"}

        if df.empty:
            return {"error": "Dataset is empty"}

        if target not in df.columns:
            return {"error": f"Target column '{target}' not found"}

        model_obj = Autodml(data=df, target=target)
        model_obj.train()

        with open("pipeline/pipeline.pkl", "wb") as f:
            pickle.dump(model_obj, f)

        return {
            "message": "Model trained successfully ",
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns": df.columns.tolist(),
            "target": target,
            "report_path": model_obj.visualizations_report,
            "input_structure": model_obj.input_features,
            "evaluation_report": model_obj.evaluation_report,
            "analysis_report": model_obj.analysis_report,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict(data: dict):
    try:
        with open("pipeline/pipeline.pkl", "rb") as f:
            model_obj = pickle.load(f)

        prediction = model_obj.predict(data)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}


@app.get("/report")
async def get_pdf():
    file_path = "data/reports/report.pdf"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, filename="Report.pdf", media_type="application/pdf"
        )
    return {"error": "File not found"}
