from data.export import AutoDMLPipeline
import pickle
import pandas as pd

model = pickle.load(open("data/model/model.pkl", "rb"))
enc = pickle.load(open("data/preprocessors/encoders.pkl", "rb"))
pca = pickle.load(open("data/preprocessors/pca.pkl", "rb"))
vec = pickle.load(open("data/preprocessors/vectorizers.pkl", "rb"))
skw = pickle.load(open("data/preprocessors/skewtransformers.pkl", "rb"))
sca = pickle.load(open("data/preprocessors/scaler.pkl", "rb"))
ft = pickle.load(open("data/preprocessors/featuretypes.pkl", "rb"))
ni = pickle.load(open("data/preprocessors/numimputer.pkl", "rb"))
ci = pickle.load(open("data/preprocessors/catimputer.pkl", "rb"))
inf = pickle.load(open("data/preprocessors/input.pkl", "rb"))

with open("data/preprocessors/textprocessor.pkl", "rb") as f:
    preprocess_text = pickle.load(f)

pipeline = AutoDMLPipeline(
    model, enc, pca, vec, skw, sca, ft, ni, ci, inf, preprocess_text
)

inp = {
    "Message": "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"
}

pred = pipeline.predict(pd.DataFrame([inp]))

print(pred)
