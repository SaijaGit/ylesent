import pandas as pd
import joblib

def run(): 
	news_data = pd.read_csv("./data/preprocessed/news_preprocessed.csv", encoding="utf-8")
	model = joblib.load("./models/model_dummy.joblib")

	X = news_data[["text"]]

	news_data["classification"] = model.predict(X)
	news_data.to_csv("./data/results/predictions.csv", index=False, encoding="utf-8")

	print("Predict.py: result = ", news_data)


