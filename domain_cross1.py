from pn_domain import *  # Import your pipeline functions
import pandas as pd
import os

def perform_cross_validation(assessments_file_path, domain_scores_df, threshold=0.3):
    """
    Perform cross-validation to assess the honesty and authenticity of author scores.
    """
    # Identify the file type and load the data
    file_suffix = os.path.splitext(assessments_file_path)[1].lower()  # Get the file extension
    if file_suffix == '.csv':
        df = pd.read_csv(assessments_file_path)
    elif file_suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(assessments_file_path)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")

    # Proceed with analysis
    adjusted_df = adjust_scores(df)  # Adjust Functioning/Wellbeing scores
    domain_avg = calculate_domain_averages(adjusted_df)  # Calculate domain averages
    results_df = cross_validation(domain_avg, domain_scores_df, threshold)
    
    # Generate summary analysis
    summary_df = generate_summary(results_df)
    
    return {
        "results_df": results_df,
        "summary_df": summary_df
     }

def adjust_scores(df):
    for index, row in df.iterrows():
        # Check if 'final_avg_score' is NaN or empty, and set it to 0 if so
        if pd.isna(row['final_avg_score']) or row['final_avg_score'] == '':
            df.at[index, 'final_avg_score'] = 0
        
        # Adjust the score for 'F' and 'W' domains
        if row['domain'] in ['F', 'W']:
            df.at[index, 'final_avg_score'] = int(4 - row['final_avg_score'])
    
    return df
def calculate_domain_averages(df):
    """
    Calculate the scaled average scores for each domain based on responses.
    """
    answered_df = df[df['final_avg_score'].notna()]
    domain_averages = answered_df.groupby('domain')['final_avg_score'].agg(['sum', 'count']).reset_index()

    # Calculate the total possible score for each domain
    domain_averages['total_possible'] = domain_averages['count'] * 4
    # Calculate the scaled average score for each domain
    domain_averages['scaled_avg'] = domain_averages['sum'] / domain_averages['total_possible']

    # Select only the necessary columns
    return domain_averages[['domain', 'scaled_avg']]

def cross_validation(assessments_df, domain_scores_df, threshold):
    """
    Compare the calculated domain scores against the expected domain scores
    and assess the honesty of the author based on a threshold.
    """
   # Create a dictionary for domain scores for easier access
    domain_scores_dict = {
    "wellbeing": domain_scores_df.loc[domain_scores_df["Domain"] == "wellbeing", "Score"].values[0],
    "functioning": domain_scores_df.loc[domain_scores_df["Domain"] == "functioning", "Score"].values[0],
    "problemsandsymptoms": domain_scores_df.loc[domain_scores_df["Domain"] == "problemsandsymptoms", "Score"].values[0],
    "risk": domain_scores_df.loc[domain_scores_df["Domain"] == "risk", "Score"].values[0],
    }

    
    # Initialize a dictionary to hold the results
    results = {
        "Domain": [],
        "Score": [],
        "Assessment Score": [],
        "Difference": [],
        "Authenticity": [],
        "Correlation": [],
    }
    
    # Define a mapping for consistent naming
    domain_mapping = {
        "wellbeing": "W",
        "functioning": "F",
        "problemsandsymptoms": "P",
        "risk": "R"
    }

    # Convert domain_scores_dict to DataFrame
    domain_scores = pd.DataFrame([domain_scores_dict])  # Wrap the dict in a list to create a single-row DataFrame
    
    # Rename columns in domain_scores based on mapping
    domain_scores.columns = [domain_mapping[col] for col in domain_scores.columns]

    # Check each domain score against the assessments
    for domain, score in domain_scores.iloc[0].items():  # Use .iloc[0] to get the values from the first row
        # Ensure that the assessment_score is retrieved correctly
        assessment_score = assessments_df.loc[assessments_df['domain'] == domain, 'scaled_avg'].values
        # Check if the assessment_score is empty
        if assessment_score.size > 0:
            assessment_score = assessment_score[0]  # Get the first value if it exists
            difference = abs(score - assessment_score)

            # Calculate the correlation for this domain
            correlation = max(0, 1 - difference)

            # Determine if the author is honest based on the threshold
            authentic = difference <= threshold
            
            # Append the results
            results["Domain"].append(domain)
            results["Score"].append(score)
            results["Assessment Score"].append(assessment_score)
            results["Difference"].append(difference)
            results["Authenticity"].append(authentic)
            results["Correlation"].append(correlation)
        else:
            # Handle the case where no assessment score is found
            results["Domain"].append(domain)
            results["Score"].append(score)
            results["Assessment Score"].append(None)  # No score found
            results["Difference"].append(None)  # No difference can be calculated
            results["Authenticity"].append(None)  # Cannot determine honesty
            results["Correlation"].append(0)  # No correlation if no score

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def generate_summary(results_df):
    """
    Generate summary statistics from the results DataFrame.
    """
    total_domains = len(results_df)
    authentic_count = results_df['Authenticity'].sum()
    average_difference = results_df['Difference'].mean()
    # Calculate percentage of domain score correlation based on correlation values
    average_correlation_percentage = results_df['Correlation'].mean() * 100
    
    # Create a summary DataFrame
    summary = {
        "Total Domains": total_domains,
        "Authentic Domain Assessments": authentic_count,
        "Percentage of Domain Score Correlation": round(average_correlation_percentage, 2),
        "Average Difference": average_difference,
    }
    
    return pd.DataFrame([summary])  # Return as a single-row DataFrame

# Execute cross-validation
# results_df, summary_df = perform_cross_validation(df, domain_scores_df)
# print("Results DataFrame:")
# print(results_df)  # Display the results
# print("\nSummary Analysis:")
# print(summary_df)  # Display the summary analysis
