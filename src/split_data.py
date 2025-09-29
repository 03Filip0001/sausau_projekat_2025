from sklearn.model_selection import train_test_split

def split_data(data=None, train_size=None, test_size=None, random_state=42):

	if data is None:
		raise Exception("[split_data] Data not provided")
	
	if train_size is None and test_size is None:
		raise Exception("[split_data] Please provide train or test size")
	
	if test_size:
		train_size = 1 - test_size

	X, y = data.drop("Churn", axis=1), data["Churn"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state)

	return X_train, X_test, y_train, y_test, X.columns.tolist()