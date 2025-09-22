import pandas as pd
from sklearn.dummy import DummyClassifier
import joblib

def run(): 
	sentiments_data = pd.read_csv("./data/preprocessed/sentiments_preprocessed.csv", encoding="utf-8")
	#print("Train.py: sentiments_data.columns = ", sentiments_data.columns)

	X = sentiments_data[["text"]]
	y = sentiments_data["sentiment"]
	
	#print("Train.py: X = ", X)
	#print("Train.py: y = ", y)

	model_dummy = DummyClassifier(strategy="most_frequent")
	model_dummy.fit(X, y)
	joblib.dump(model_dummy, "models/model_dummy.joblib")

	print("Train.py: Classifier created")