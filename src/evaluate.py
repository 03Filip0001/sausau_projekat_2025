from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report

from src.utils import _print_msg

def evaluate(model_name=None, y_pred=None, y_test=None, labels=None, model=None, feature_names=None, most_important=False):
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	cm = confusion_matrix(y_test, y_pred)

	report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)

	_print_msg(msg=f"Results for {model_name}:")
	_print_msg(msg="     Accuracy:"+str(accuracy))
	_print_msg(msg="     Precision:"+str(precision))
	_print_msg(msg="     Recall:"+str(recall))
	_print_msg(msg="     F1-Score:"+str(f1), nl=True, sep=True)

	_print_msg(msg="Confusion matrix:")
	_print_msg(msg=str(cm), nl=True, sep=True)

	_print_msg(msg="Classification report:")
	_print_msg(msg=str(report), nl=True, sep=True)

	if most_important:
		_print_msg(msg="Feature importances in model:")
		if not hasattr(model, "feature_importances_"):
			return
		
		importances = model.feature_importances_
		feature_importances = {}
		for name, importance in zip(feature_names, importances):
			_print_msg(msg=f"     {name}: {importance:.5f}")
			feature_importances[name] = importance
		_print_msg(msg="", nl=True, sep=True)

		most_important_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]
		_print_msg(msg="Top 5 most important:")
		for fe, val in most_important_features:
			_print_msg(msg=f"     {fe} with importance {val:.5f}")
