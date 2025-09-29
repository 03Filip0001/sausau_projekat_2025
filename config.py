from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DEFAULT_MODEL_CLASS = GradientBoostingClassifier
DEFAILT_MODEL_NAME = "Gradient Boosting"

# DEFAULT_MODEL_CLASS = LogisticRegression
# DEFAILT_MODEL_NAME = "Logistic Regression"

# DEFAULT_MODEL_CLASS = RandomForestClassifier
# DEFAILT_MODEL_NAME = "Forest Classifier"

DEFAULT_DATA_PATH = "data/telco_data.csv"
DEFAULT_PARAMS_PATH = "data/params.json"

# Models used for this project
MODEL_NAMES = {
	"Logistic Regression": LogisticRegression,
	"Forest Classifier": RandomForestClassifier,
	"Gradient Boosting": GradientBoostingClassifier
}