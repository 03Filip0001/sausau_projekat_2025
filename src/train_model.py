from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils import _print_msg

param_lr = {
	"C": [0.01, 0.1, 1, 10],
	"solver": ["lbfgs", "liblinear"],
	"max_iter": [100, 200, 500, 1000, 2000],
	"class_weight": ["balanced"]
}

param_rf = {
	"n_estimators": [50, 75, 125, 200],
	"max_depth": [None, 5],
	"min_samples_split": [2, 5, 10],
	"class_weight": ["balanced"]
}

param_gb = {
	"n_estimators": [50, 75, 125, 200],
	"learning_rate": [0.01, 0.1, 0.2],
	"max_depth": [3, 5]
}

def _train_model(X_train=None, y_train=None, model_class=None, model_name=""):

	if model_class == LogisticRegression:
		grid = RandomizedSearchCV(model_class(), param_distributions=param_lr, n_iter=20, cv=5, scoring='accuracy', random_state=42)
	elif model_class == RandomForestClassifier:
		grid = RandomizedSearchCV(model_class(), param_distributions=param_rf, n_iter=15, cv=5, scoring='accuracy', random_state=42)
	elif model_class == GradientBoostingClassifier:
		grid = RandomizedSearchCV(model_class(), param_distributions=param_gb, n_iter=15, cv=5, scoring='accuracy', random_state=42)

	else:
		raise Exception("Please provide correct sklearn model")
	
	grid.fit(X_train, y_train)

	return grid.best_params_

def train_model(X_train=None, y_train=None, model_class=None, model_name=""):

	_print_msg(msg="Training model "+model_name)

	best_param = _train_model(X_train=X_train, y_train=y_train, model_class=model_class, model_name=model_name)
	model = model_class(**best_param)

	model.fit(X_train, y_train)

	return model