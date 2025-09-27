print("\n\n######################################")
print("Projekat IV: Telco Customer Churn")
print("######################################\n\n")

from src.load_data import load_data, parse_data
from src.split_data import split_data

# Loading data
data = load_data(fName="data/telco_data.csv")

# Preproccess data
data, encoders = parse_data(data=data)

# Spliting data
X_train, X_test, y_train, y_test = split_data(data=data, train_size=0.7)

print(data.head())