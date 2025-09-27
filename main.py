print("\n\n######################################")
print("Projekat IV: Telco Customer Churn")
print("######################################\n\n")

from src.load_data import load_data, parse_data
from src.split_data import split_data
from src.train_model import train_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score

model_names = {
	"Logistic Regression" : LogisticRegression,
	"Random Forest Classifier": RandomForestClassifier,
	"Gradient Boosting Classifier": GradientBoostingClassifier
}

# Loading data
data = load_data(fName="data/telco_data.csv")

# Preproccess data
data, encoders = parse_data(data=data)

# Spliting data
X_train, X_test, y_train, y_test = split_data(data=data, train_size=0.7)

# Training models
models = {}
for name, model in model_names.items():
	models[name] = train_model(X_train=X_train, y_train=y_train, model_class=model, model_name=name)

# Predicting results
for name, model in models.items():
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy for model", name, accuracy)