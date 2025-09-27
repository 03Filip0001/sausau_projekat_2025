from src.utils import _print_msg

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

_max_iter = [100, 250, 500, 1000, 2000]
_n_estimators = [25, 50, 100, 150, 300]

configuration_logistic = {
	"max_iter": _max_iter[0],
	"solver": "lbfgs",
	"class_weight": "balanced",
	"random_state": 0
}

configuration_forest = {
	"n_estimators": _n_estimators[0],
	"class_weight": "balanced",
	"random_state": 0
}

configuration_gradient = {
	"n_estimators": _n_estimators[0],
	"learning_rate": 0.1,
	"random_state": 0
}

def _train_model(X_train=None, y_train=None, model_class=None, model_name="", random_state=42):
	_print_msg(msg="Training model "+model_name)
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

def train_model(X_train=None, y_train=None, model_class=None, model_name="", random_state=42):
	if (X_train is None) or (y_train is None):
		raise Exception("[train_model] Provide X_train and y_train values")

	if model_class is None:
		raise Exception("[train_model] Provide model")
	
	model = {}

	if model_class == LogisticRegression:
		for iter in _max_iter:
			configuration_logistic["max_iter"] = iter
			model[iter] = _train_model(X_train=X_train, y_train=y_train, model_class=model_class, model_name=model_name, random_state=random_state)

	else:
		for n in _n_estimators:
			configuration_forest["n_estimators"] = n
			configuration_gradient["n_estimators"] = n
			model[n] = _train_model(X_train=X_train, y_train=y_train, model_class=model_class, model_name=model_name, random_state=random_state)

	return model