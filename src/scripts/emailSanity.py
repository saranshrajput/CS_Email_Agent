import pandas as pd

df = pd.read_csv("../../data/emails.csv")

# Peek at the first few rows
print(df.head())

# Count per category
print(df['label'].value_counts())

# Approximate short vs long split
df['length'] = df['email'].apply(len)
print(df['length'].describe())
