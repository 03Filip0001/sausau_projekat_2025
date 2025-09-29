from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report

def evaluate(model_name=None, y_pred=None, y_test=None, labels=None, model=None, feature_names=None, most_important=False):
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	cm = confusion_matrix(y_test, y_pred)

	report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)

	print(f"Results for {model_name}:")
	print("Accuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)
	print("F1-Score:", f1)

	print(report)
	print(cm)

	if most_important:
		pass
		if not hasattr(model, "feature_importances_"):
			return
		
		importances = model.feature_importances_
		feature_importances = {}
		for name, importnace in zip(feature_names, importances):
			feature_importances[name] = importnace

		print(feature_importances)

		most_important_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]
		print(most_important_features)
