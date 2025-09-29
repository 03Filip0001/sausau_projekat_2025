from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils import _print_msg
from config import *

def _train_model(X_train=None, y_train=None, model_class=None, model_name=""):

	if model_class == LogisticRegression:
		grid = RandomizedSearchCV(model_class(), param_distributions=DEFAULT_PARAMS_LR, 
							n_iter=20, cv=5, scoring='f1', random_state=42, verbose=1)
	elif model_class == RandomForestClassifier:
		grid = RandomizedSearchCV(model_class(), param_distributions=DEFAULT_PARAMS_RF,
							 n_iter=20, cv=5, scoring='f1', random_state=42, verbose=1)
	elif model_class == GradientBoostingClassifier:
		grid = RandomizedSearchCV(model_class(), param_distributions=DEFAULT_PARAMS_GB, 
							n_iter=20, cv=5, scoring='f1', random_state=42, verbose=1)

	else:
		raise Exception("Please provide correct sklearn model")
	
	grid.fit(X_train, y_train)

	return grid.best_params_, grid

def train_model(X_train=None, y_train=None, model_class=DEFAULT_MODEL_CLASS, model_name=DEFAILT_MODEL_NAME, models_params=None, feature_names=None):
	
	if X_train is None: raise ValueError("Please provide X_train data")
	if y_train is None: raise ValueError("Please provide y_train data")
	if models_params is None: raise ValueError("Please provide model parameters")
	if feature_names is None: raise ValueError("Please provide feature_names data")
	if model_name not in MODEL_NAMES or model_class != MODEL_NAMES[model_name]: raise Exception("Unknown model !")

	params = models_params[model_name]
	_print_msg(msg=f"Creating model \'{model_name}\' with parameters: "+str(params)+"...")
	model = model_class(**params)

	_print_msg(msg="Training model...")
	model.fit(X_train, y_train)

	_print_msg(msg="Model feature importances:")
	if hasattr(model, "feature_importances_"):
		importances = model.feature_importances_
		
		for name, importance in zip(feature_names, importances):
			_print_msg(msg=f"     {name}: {importance:.5f}")
	elif hasattr(model, "coef_"):
		importances = model.coef_[0]

		for name, importance in zip(feature_names, importances):
			_print_msg(msg=f"     {name}: {importance:.5f}")

	return model, model_name
	

def find_best_params(X_train=None, y_train=None, model_class=None, model_name=""):
	
	_print_msg(msg="Finding params for model "+model_name)
	best_param, model = _train_model(X_train=X_train, y_train=y_train, model_class=model_class, model_name=model_name)
	_print_msg(msg="Best params for model: "+str(best_param), nl=True, sep=True)

	return best_param, model