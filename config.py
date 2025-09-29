from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

"""
	Default model which will be run when program starts
	In order to run another model with best trained parameters uncomment below variables in group
"""
# DEFAULT_MODEL_CLASS = GradientBoostingClassifier
# DEFAILT_MODEL_NAME = "Gradient Boosting"

# DEFAULT_MODEL_CLASS = LogisticRegression
# DEFAILT_MODEL_NAME = "Logistic Regression"

DEFAULT_MODEL_CLASS = RandomForestClassifier
DEFAILT_MODEL_NAME = "Forest Classifier"


"""
	Default files location
	DO NOT CHANGE!
"""
DEFAULT_DATA_PATH = "data/telco_data.csv"
DEFAULT_PARAMS_PATH = "data/params.json"

""" 
	Models used for this project
"""
MODEL_NAMES = {
	"Logistic Regression": LogisticRegression,
	"Forest Classifier": RandomForestClassifier,
	"Gradient Boosting": GradientBoostingClassifier
}
DEFAULT_PARAMS_LR = {
	"C": [0.01, 0.1, 1, 10, 15],
	"solver": ["lbfgs", "liblinear"],
	"max_iter": [100, 200, 500, 1000, 2000, 2500],
	"class_weight": ["balanced"]
}

DEFAULT_PARAMS_RF = {
	"n_estimators": [50, 75, 125, 150, 200, 300],
	"max_depth": [None, 5],
	"min_samples_split": [2, 5, 10, 20],
	"class_weight": ["balanced"]
}

DEFAULT_PARAMS_GB = {
	"n_estimators": [50, 75, 125, 150, 200, 300],
	"learning_rate": [0.01, 0.1, 0.2],
	"max_depth": [3, 5, 10]
}