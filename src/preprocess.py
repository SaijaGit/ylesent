import pandas as pd

def run(): 
	news_data = pd.read_csv("./data/raw/news.csv", encoding="utf-8")
	sentiments_data = pd.read_csv("./data/raw/sentiments.csv", encoding="utf-8")

	news_data = news_data.drop(columns=["headline"])

	print("Preprocess.py: sentiments_data = ", sentiments_data)

	news_data.to_csv("./data/preprocessed/news_preprocessed.csv", index=False)
	sentiments_data.to_csv("./data/preprocessed/sentiments_preprocessed.csv", index=False)