print("\n\n######################################")
print("Projekat IV: Telco Customer Churn")
print("######################################\n\n")

from src.load_data import load_data

data = load_data(fName="data/telco_data.csv")

print(data.head())