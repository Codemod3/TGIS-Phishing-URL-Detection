import pandas as pd

# Load your training data
df = pd.read_parquet("data/processed/train.parquet")

# Filter only the SAFE sites (label == 0)
safe_sites = df[df['label'] == 0]

# Count how many safe sites have a domain age of exactly 0
zero_age_safe_count = len(safe_sites[safe_sites['domain_age_days'] == 0])
total_safe = len(safe_sites)

print(f"Total Safe Sites: {total_safe}")
print(f"Safe Sites with 0 Age: {zero_age_safe_count}")
print(f"Percentage Poisoned: {(zero_age_safe_count / total_safe) * 100:.2f}%")