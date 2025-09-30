import pandas as pd
import joblib
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


def run(): 
	print("Train.py start!")
	training_data = pd.read_csv("./data/preprocessed/training_preprocessed.csv", encoding="utf-8")
	print("Train.py: Training data loaded, training_data.columns = ", training_data.columns)

	X = training_data["text_for_model"].astype(str)
	y = training_data["label"]
	
	print("Train.py: X = ", X.head(5))
	print("Train.py: y = ", y.head(5))

	# Split data for training and testing
	test_size = 0.2
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

	# Build pipeline = model
	pipe = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase = False)),
    ('logreg', LogisticRegression(solver = 'saga', max_iter=100))
	])

	# Train the model
	pipe.fit(X_train, y_train)
	print("Train.py: Model created")

	# Save the model
	joblib.dump(pipe, "models/model_logreg.joblib")

	# Make predictions using the testing set
	y_pred = pipe.predict(X_test)

	# The mean squared error
	print('Train.py: Mean squared error: %.2f',	mean_squared_error(y_test, y_pred))
	# The coefficient of determination: 1 is perfect prediction
	print('Train.py: Coefficient of determination: %.2f', r2_score(y_test, y_pred))

