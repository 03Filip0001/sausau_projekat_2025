from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils import _print_msg
from config import DEFAILT_MODEL_NAME, DEFAULT_MODEL_CLASS

param_lr = {
	"C": [0.01, 0.1, 1, 10, 15],
	"solver": ["lbfgs", "liblinear"],
	"max_iter": [100, 200, 500, 1000, 2000, 2500],
	"class_weight": ["balanced"]
}

param_rf = {
	"n_estimators": [50, 75, 125, 150, 200, 300],
	"max_depth": [None, 5],
	"min_samples_split": [2, 5, 10, 20],
	"class_weight": ["balanced"]
}

param_gb = {
	"n_estimators": [50, 75, 125, 150, 200, 300],
	"learning_rate": [0.01, 0.1, 0.2],
	"max_depth": [3, 5, 10]
}

def _train_model(X_train=None, y_train=None, model_class=None, model_name=""):

	if model_class == LogisticRegression:
		grid = RandomizedSearchCV(model_class(), param_distributions=param_lr, n_iter=20, cv=5, scoring='f1', random_state=42)
	elif model_class == RandomForestClassifier:
		grid = RandomizedSearchCV(model_class(), param_distributions=param_rf, n_iter=20, cv=5, scoring='f1', random_state=42)
	elif model_class == GradientBoostingClassifier:
		grid = RandomizedSearchCV(model_class(), param_distributions=param_gb, n_iter=20, cv=5, scoring='f1', random_state=42)

	else:
		raise Exception("Please provide correct sklearn model")
	
	grid.fit(X_train, y_train)

	return grid.best_params_, grid

def train_model(X_train=None, y_train=None, model_class=DEFAULT_MODEL_CLASS, model_name=DEFAILT_MODEL_NAME, models_params=None, feature_names=None):
	
	params = models_params[model_name]
	model = model_class(**params)

	model.fit(X_train, y_train)

	if hasattr(model, "feature_importances_"):
		importances = model.feature_importances_
		
		for name, importance in zip(feature_names, importances):
			print(f" {name}: {importance:.3f}")
	elif hasattr(model, "coef_"):
		importances = model.coef_[0]

		for name, importance in zip(feature_names, importances):
			print(f" {name}: {importance:.3f}")

	return model, model_name
	

def find_best_params(X_train=None, y_train=None, model_class=None, model_name=""):
	_print_msg(msg="Finding params for model "+model_name)
	best_param, model = _train_model(X_train=X_train, y_train=y_train, model_class=model_class, model_name=model_name)

	_print_msg(msg="Best params for model: ")
	print(best_param)

	return best_param, model