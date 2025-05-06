# eda.py

import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(bank_df):
    print("=== Exploratory Data Analysis ===\n")
    
    # Checking unique values
    print("Unique Transactions:", bank_df['TransactionID'].nunique())
    print("Unique Customers:", bank_df['CustomerID'].nunique())
    
    # Unique transaction dates
    unique_dates = bank_df['TransactionDate'].unique()
    print("\nUnique Transaction Dates:", unique_dates)

    # Distribution across transaction dates
    plt.figure(figsize=(10, 6))
    sns.histplot(bank_df['TransactionDate'], bins=3, kde=False, color='skyblue')
    plt.title('Transaction Date Distribution')
    plt.xlabel('Transaction Date')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nNote: Dataset covers ~3 months (Aug–Oct 2016), which explains low customer frequency despite high total transactions.")

    # Pie chart of gender distribution
    plt.figure(figsize=(8, 6))
    gender_count = bank_df['CustGender'].value_counts()
    plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=100)
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.show()
