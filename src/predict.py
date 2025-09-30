import pandas as pd
import joblib

def run(): 
	news_data = pd.read_csv("./data/preprocessed/cleaned_news.csv", encoding="utf-8", header = None, names = ["date", "text"])
	model = joblib.load("./models/model_logreg.joblib")

	X = news_data["text"].astype(str)

	news_data["classification"] = model.predict(X)
	news_data.to_csv("./data/results/predictions.csv", index=False, encoding="utf-8")

	print("Predict.py: result = ", news_data)


