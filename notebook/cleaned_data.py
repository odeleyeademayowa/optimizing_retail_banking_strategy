import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing code...

def clean_data(bank_df):
    """
    Function that runs all the data cleaning and preprocessing steps
    """
    # Remove duplicates
    duplicate_count = bank_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    # Rename columns
    bank_df.rename(columns={'TransactionAmount (INR)': 'TransactionAmount'}, inplace=True)
    
    # Convert date columns to datetime
    bank_df['TransactionDate'] = pd.to_datetime(bank_df['TransactionDate'], dayfirst=True, errors='coerce')
    bank_df['CustomerDOB'] = pd.to_datetime(bank_df['CustomerDOB'], errors='coerce')
    
    # Correct Date of Birth issues (after 100 years or invalid DOBs)
    mask = bank_df['CustomerDOB'] > bank_df['TransactionDate']
    bank_df.loc[mask, 'CustomerDOB'] = bank_df.loc[mask, 'CustomerDOB'] - pd.DateOffset(years=100)
    
    # Calculate age
    reference_date = bank_df['TransactionDate'].max()
    bank_df['Age'] = (reference_date - bank_df['CustomerDOB']).dt.days // 365
    
    # Clean extreme DOB values (before 1900 or after 2016)
    start_date = pd.to_datetime('1900-01-01')
    end_date = pd.to_datetime('2016-10-21')
    valid_dob_median = bank_df[(bank_df['CustomerDOB'] >= start_date) & (bank_df['CustomerDOB'] <= end_date)]['CustomerDOB'].median()
    bank_df.loc[(bank_df['CustomerDOB'] < start_date) | (bank_df['CustomerDOB'] > end_date), 'CustomerDOB'] = valid_dob_median
    
    # Final cleaning steps (removing zero-transaction rows)
    bank_df = bank_df[bank_df['TransactionAmount'] != 0]
    
    # Display final cleaned dataset info
    print("Data Cleaning Completed!")
    print(bank_df.info())
    return bank_df
