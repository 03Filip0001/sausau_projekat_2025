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