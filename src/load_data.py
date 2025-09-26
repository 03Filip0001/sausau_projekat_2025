import os
import numpy as np
import pandas as pd

from src.utils import _print_msg
from sklearn.preprocessing import LabelEncoder, StandardScaler

def _print_column(msg="Column: ", col=None, sep=False, nl=False):
	if col is None:
		return
	
	if type(col) == list:
		msg = msg + ", ".join(col)
	else:
		msg = msg + str(col)
	_print_msg(msg=msg, sep=sep, nl=nl)

def _print_dropping_column(msg="Dropping columns: ", col=None, sep=False, nl=False):
	_print_column(msg=msg, col=col, sep=sep, nl=nl)

def load_data(fName=None):
	if not fName:
		raise Exception("Provide data file name")
	
	_print_msg(msg="Trying to read data from file: "+fName+"...")
	
	if not os.path.exists(fName):
		raise Exception("File does not exist")	

	# Reading CSV file for data
	data = pd.read_csv(fName, na_values=["", " "])

	_print_msg(msg="Done reading data.", sep=True)
	return data

def fill_data(data=None, col=None, str_col=None):
	if data is None or col is None or str_col is None:
		raise Exception("[fill_data]: \"Data, col or str_col not provided\"")

	if col in str_col:
		_print_msg(msg="Column type is text")
		most_freq = data[col].mode()[0]
		_print_column(msg="Most frequent value is", col=most_freq)
		_print_msg(msg="Filling empty rows with that value...", nl=True)
		data.loc[data[col].isnull() | (data[col] == " "), col] = most_freq

	else:
		_print_msg(msg="Column type should be number")
		_print_msg(msg="Converting to number...")
		if pd.api.types.is_string_dtype(data[col]):
			data[col] = pd.to_numeric(data[col], errors="coerce")

		_print_msg(msg="Finding mean value for row...")
		mean_value = data[col].mean()

		_print_column(msg="Mean value is", col=mean_value)
		_print_msg("Filling empty rows with that value...")
		data[col] = data[col].fillna(mean_value)

	_print_column(msg="Done filling column ", col=col, sep=True, nl=True)
		
def parse_data(data=None):
	if data is None or data.empty:
		raise Exception("[parse_data]: \"Data not provided\"")
	_print_msg("Parsing data for model...")

	# Which columns to drop - model doesn't need them or not enough data in them
	columns_drop = ["customerID"]

	# String columns in dataset
	string_columns = [
		"gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", 
		"OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
		"StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "customerID"]

	# Calculating the minimum number of rows in each column needed for model (80%)
	num_records = len(data)
	missing_num_max = num_records * 0.2

	_print_msg(msg="Filling the empty rows and dropping unuseful columns...", sep=True, nl=True)
	for col in data.columns:
		missing_rows = data[col].isnull().sum() + (data[col] == " ").sum()
		if missing_rows > missing_num_max:
			_print_dropping_column(col=col)
			columns_drop.append(col)
		elif missing_rows:
			_print_column(msg="Filling empty rows in column", col=col)
			fill_data(data=data, col=col, str_col=string_columns)
			
	columns_drop = list(set(columns_drop))
	_print_dropping_column(col=columns_drop, sep=True, nl=True)
	data.drop(columns=columns_drop, inplace=True)

	_print_msg(msg="Converting \"Yes/No\" columns to 1/0...")
	binary_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
	_print_column(col=binary_columns, nl=True)
	for col in binary_columns:
		if col in data.columns:
			data[col] = data[col].map({"Yes": 1, "No": 0})

	_print_msg("Converting \"Yes/No/Other\" columns to 1/2/0...")
	ternary_columns = ["MultipleLines", "OnlineSecurity", "OnlineBackup", 
					"DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
	_print_column(col=ternary_columns, sep=True, nl=True)
	for col in ternary_columns:
		if col in data.columns:
			data[col] = data[col].map({"No": 1, "Yes": 2}).fillna(0).astype(int)

	_print_msg("Encoding string columns...")
	other_columns = ["gender", "InternetService", "Contract", "PaymentMethod"]
	_print_column(col=other_columns, nl=True)

	encoders = {}
	for col in other_columns:
		if col in data.columns:
			lEncoder = LabelEncoder()
			data[col] = lEncoder.fit_transform(data[col])
			encoders[col] = lEncoder

	_print_msg("Standarding values...")
	standard_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
	_print_column(col=other_columns, sep=True, nl=True)

	scaler = StandardScaler()
	data["TotalCharges"] = np.log1p(data["TotalCharges"])
	data[standard_columns] = scaler.fit_transform(data[standard_columns])

	return data, encoders