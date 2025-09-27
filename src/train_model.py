from src.utils import _print_msg

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

configuration_logistic = {
	"max_iter": 1000,
	"solver": "lbfgs",
	"class_weight": "balanced",
	"random_state": 0
}

configuration_forest = {
	"n_estimators": 100,
	"class_weight": "balanced",
	"random_state": 0
}

configuration_gradient = {
	"n_estimators": 100,
	"learning_rate": 0.1,
	"random_state": 0
}

def train_model(X_train=None, y_train=None, model_class=None, model_name="", random_state=42):
	_print_msg(msg="Training model "+model_name)

	if (X_train is None) or (y_train is None):
		raise Exception("[train_model] Provide X_train and y_train values")

	if model_class is None:
		raise Exception("[train_model] Provide model")
	
	_print_msg(msg="With configuration: ")

	if model_class == LogisticRegression:
		configuration_logistic["random_state"] = random_state
		_print_msg(msg=str(configuration_logistic))
		model = model_class(**configuration_logistic)

	elif model_class == RandomForestClassifier:
		configuration_forest["random_state"] = random_state
		_print_msg(msg=str(configuration_forest))
		model = model_class(**configuration_forest)

	elif model_class == GradientBoostingClassifier:
		configuration_gradient["random_state"] = random_state
		_print_msg(msg=str(configuration_gradient))
		model = model_class(**configuration_gradient)

	else:
		raise Exception("[train_model] Please provide correct model")
	
	_print_msg(msg="Configuration done")
	_print_msg(msg="Training model...")

	model = model.fit(X_train, y_train)

	_print_msg(msg="Done training model "+model_name, nl=True, sep=True)

	return model