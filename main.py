# main.py

from data_loading import load_data, inspect_data, check_unique_values, check_missing_values, split_features
from cleaned_data import clean_data  # Assuming you've wrapped your cleaning steps in this function
from eda import run_eda  # Import the EDA function
from rfm_analysis import calculate_rfm, calculate_rfm_scores, cluster_rfm, visualize_clusters, save_rfm_to_csv  # Import RFM analysis functions

def main():
    # Load the dataset
    filepath = "../data/bank_data_C.csv"
    bank_df = load_data(filepath)

    # Inspect raw data
    inspect_data(bank_df)

    # Check unique and missing values
    check_unique_values(bank_df)
    check_missing_values(bank_df)

    # Clean the data
    print("Starting Data Cleaning Process...")
    clean_data(bank_df)

    # Re-check the data after cleaning
    inspect_data(bank_df)

    # Split features
    numerical_features, categorical_features = split_features(bank_df)
    print("\nNumerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    # Run Exploratory Data Analysis
    run_eda(bank_df)

    # Perform RFM Analysis
    print("\nStarting RFM Analysis...")
    
    # Calculate RFM metrics
    rfm = calculate_rfm(bank_df)
    
    # Calculate RFM scores and add them to the dataframe
    rfm, r_quartiles, f_quartiles, m_quartiles = calculate_rfm_scores(rfm)
    
    # Perform clustering on the RFM data
    rfm = cluster_rfm(rfm)
    
    # Visualize clusters and metrics
    visualize_clusters(rfm)
    
    # Save RFM results to a CSV file
    save_rfm_to_csv(rfm)
    
    # Optionally print the RFM results
    print("\nRFM Analysis Results:")
    print(rfm.head())

if __name__ == "__main__":
    main()

