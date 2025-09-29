print("\n\n######################################")
print("Projekat IV: Telco Customer Churn")
print("######################################\n\n")

from src.load_data import load_data, parse_data
from src.split_data import split_data

from src.train_model import train_model, find_best_params
from src.evaluate import evaluate

from src.utils import _print_msg

from config import *

import sys
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score, recall_score

_TRAINING_MODELS = False

model_names = {
	"Logistic Regression": LogisticRegression,
	"Forest Classifier": RandomForestClassifier,
	"Gradient Boosting": GradientBoostingClassifier
}

if len(sys.argv) > 1:
	argv = sys.argv[1]
	_TRAINING_MODELS = True
	_print_msg(msg="ENTERING TRAINING MODELS FUNCTION", nl=True, sep=True)
else:
	_TRAINING_MODELS = False
	_print_msg(msg="ENTERING MODEL EVALUATION FUNCTION", nl=True, sep=True)


# Loading data
data = load_data(fName=DEFAULT_DATA_PATH)

# Preproccess data
data, encoders = parse_data(data=data)

# Spliting data
X_train, X_test, y_train, y_test, features = split_data(data=data, train_size=0.7)
models_param = {}

# sacuvaj sve modele i odredi koji je najbolji
#u main pozovi samo taj najbolji
if _TRAINING_MODELS:
	for name, model in model_names.items():
		models_param[name], model = find_best_params(X_train=X_train, y_train=y_train, model_class=model, model_name=name)

		y_pred = model.predict(X_test)
		evaluate(model_name=name, y_pred=y_pred, y_test=y_test, labels=["No Churn", "Churn"])

	with open(DEFAULT_PARAMS_PATH, "w") as f:
		json.dump(models_param, f)

else:
	if os.path.exists(DEFAULT_PARAMS_PATH):
		with open(DEFAULT_PARAMS_PATH, "r") as f:
			models_param = json.load(f)

	model, model_name = train_model(X_train=X_train, y_train=y_train, models_params=models_param, feature_names=features)
	y_pred = model.predict(X_test)

	evaluate(model_name=model_name, y_pred=y_pred, y_test=y_test, labels=["No Churn", "Churn"], model=model, feature_names=features, most_important=True)
	
# for name, model in models.items():
# 	y_pred = model.predict(X_test)

# 	print(f"Results for {name}:")
# 	print("Accuracy:", accuracy_score(y_test, y_pred))
# 	print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# 	print("Classification Report:\n", classification_report(y_test, y_pred))