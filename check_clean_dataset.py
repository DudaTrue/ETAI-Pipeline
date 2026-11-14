import yaml
from src.data import load_data
from src.preprocessing import clean_dataset

# Load your Adult config and data
with open("config_adult.yaml", "r") as f:
    config = yaml.safe_load(f)

df = load_data("data/adult_train.csv")
df_clean = clean_dataset(df, config["diagnostics"])

# Check the missing counts for the specific columns
for col in ["workclass", "occupation", "native_country"]:
    print(f"\n--- {col} ---")
    print(df_clean[col].value_counts(dropna=False))