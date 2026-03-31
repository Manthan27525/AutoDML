from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import os
from autodml import Autodml
import pickle
import chardet
from fastapi.middleware.cors import CORSMiddleware


def detect_encoding(file_path, n_bytes=10000):
    with open(file_path, "rb") as f:
        raw_data = f.read(n_bytes)
    result = chardet.detect(raw_data)
    return result["encoding"] or "utf-8"


def load_dataset(file_path: str) -> pd.DataFrame:
    if not file_path or not isinstance(file_path, str):
        raise ValueError("Invalid file path")

    if not (file_path.endswith(".csv") or file_path.endswith(".xlsx")):
        raise ValueError("Only CSV and XLSX files are supported")

    if file_path.endswith(".csv"):
        encoding = detect_encoding(file_path)

        try:
            df = pd.read_csv(
                file_path, encoding=encoding, engine="python", on_bad_lines="skip"
            )
        except:
            df = pd.read_csv(
                file_path, encoding="latin1", engine="python", on_bad_lines="skip"
            )

    else:
        df = pd.read_excel(file_path, engine="openpyxl")

    if df is None or df.empty:
        raise ValueError("Dataset is empty or could not be loaded")

    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns")

    return df


app = FastAPI(title="AutoDML API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

        for f in os.listdir("uploads"):
            os.remove(os.path.join("uploads", f))

        ext = file.filename.split(".")[-1]
        file_path = f"uploads/data.{ext}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".csv"):
            df = load_dataset(file_path)
        elif file.filename.endswith(".xlsx"):
            df = load_dataset(file_path)
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
