import os
import pandas as pd

def load_data(fName=None):
	if not fName:
		raise Exception("Provide data file name")
	
	if not os.path.exists(fName):
		raise Exception("File does not exist")	

	print("File exists")

	data = pd.read_csv(fName)

	return data