print("\n\n######################################")
print("Projekat IV: Telco Customer Churn")
print("######################################\n\n")

from src.load_data import load_data, parse_data
from src.split_data import split_data

from src.train_model import train_model, find_best_params
from src.evaluate import evaluate

from src.utils import _print_msg
from config import *

import sys, json, os

# Chosing either to train or run model with saved parameters
_TRAINING_MODELS = False

if len(sys.argv) > 1:
	argv = sys.argv[1]
	_TRAINING_MODELS = True
	_print_msg(msg="ENTERING TRAINING MODELS FUNCTION", nl=True, sep=True)
	
else:
	_TRAINING_MODELS = False
	_print_msg(msg="ENTERING MODEL EVALUATION FUNCTION", nl=True, sep=True)


# Loading data
data = load_data(fName=DEFAULT_DATA_PATH)

# Preprocess data
data, encoders = parse_data(data=data)

# Spliting data
X_train, X_test, y_train, y_test, features = split_data(data=data, train_size=0.7)

# Running models
models_param = {}
if _TRAINING_MODELS:
	# Finding best params for models
	for name, model in MODEL_NAMES.items():
		models_param[name], model = find_best_params(X_train=X_train, y_train=y_train, model_class=model, model_name=name)

		# Priting each model results
		y_pred = model.predict(X_test)
		evaluate(model_name=name, y_pred=y_pred, y_test=y_test, labels=["No Churn", "Churn"])

	_print_msg(msg="Saving model params to file "+DEFAULT_PARAMS_PATH+"...")
	with open(DEFAULT_PARAMS_PATH, "w") as f:
		json.dump(models_param, f)

	_print_msg(msg="Done saving params.", nl=True, sep=True)

else:
	_print_msg(msg="Loading model params from file "+DEFAULT_PARAMS_PATH+"...")
	if os.path.exists(DEFAULT_PARAMS_PATH):
		with open(DEFAULT_PARAMS_PATH, "r") as f:
			models_param = json.load(f)

	_print_msg(msg="Done loading params.", nl=True, sep=True)

	# Trainig model with loaded params
	model, model_name = train_model(X_train=X_train, y_train=y_train, models_params=models_param, feature_names=features)
	y_pred = model.predict(X_test)

	# Evaluating model
	evaluate(model_name=model_name, y_pred=y_pred, y_test=y_test, labels=["No Churn", "Churn"], 
		  model=model, feature_names=features, most_important=True)