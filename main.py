print("\n\n######################################")
print("Projekat IV: Telco Customer Churn")
print("######################################\n\n")

from src.load_data import load_data, parse_data

# Loading data
data = load_data(fName="data/telco_data.csv")

# Preproccess data
data, encoders = parse_data(data=data)

print(data.head())