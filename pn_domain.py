#For calculation domain scores - spliting the sentences under wellbeing and functioning domains to positive and negative.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function for training and saving models
def train_and_save_models(category_data, save_dir='domain_positive_negative_models', test_size=0.2, random_state=42):
    """
    Trains a Naive Bayes model for each category in the dataset and saves the model and vectorizer in the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    models = {}
    vectorizers = {}

    for category, df in category_data.items():
        print(f"Training model for category: {category}")
        model_filename = os.path.join(save_dir, f'{category}_model.pkl')
        vectorizer_filename = os.path.join(save_dir, f'{category}_vectorizer.pkl')

        # Check if model and vectorizer already exist
        if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
            print(f"Loading existing model and vectorizer for category: {category}")
            models[category] = joblib.load(model_filename)
            vectorizers[category] = joblib.load(vectorizer_filename)
            continue
        
        print(f"Training model for category: {category}")

        X = df['sentence']
        y = df[category]

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Vectorize the text
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_val_vectorized = vectorizer.transform(X_val)

        # Train the Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val_vectorized)
        print(f"Accuracy for {category}: {accuracy_score(y_val, y_pred)}")
        print(f"Classification Report for {category}:\n{classification_report(y_val, y_pred)}")

        # Save the model and vectorizer
        model_filename = os.path.join(save_dir, f'{category}_model.pkl')
        vectorizer_filename = os.path.join(save_dir, f'{category}_vectorizer.pkl')

        joblib.dump(model, model_filename)
        joblib.dump(vectorizer, vectorizer_filename)

        models[category] = model
        vectorizers[category] = vectorizer

    return models, vectorizers

# Function to filter sentences by domain
def filter_sentences_by_domain(predictions_filepath, label_f_col='predicted_label_f', label_w_col='predicted_label_w',label_ps_col='predicted_label_ps', label_r_col='predicted_label_rv'):
    """
    Filters sentences based on predicted domain labels from a given Excel file.
    """
    df = pd.read_excel(predictions_filepath)
    filtered_sentences_f_only = df[df[label_f_col] == 'f'][['questions', label_f_col]]
    filtered_sentences_w_only = df[df[label_w_col] == 'w'][['questions', label_w_col]]
    filtered_sentences_ps_only = df[df[label_ps_col] == 'ps'][['questions', label_ps_col, 'probability_ps']]
    filtered_sentences_rv_only = df[df[label_r_col] == 'rv'][['questions', label_r_col, 'probability_rv']]
    return filtered_sentences_f_only, filtered_sentences_w_only, filtered_sentences_ps_only, filtered_sentences_rv_only

# Function to classify sentences for a specific category
def classify_sentences(df, category, models_dir='domain_positive_negative_models'):
    """
    Classifies sentences based on pre-trained model and vectorizer for the specific category.
    """
    model_filename = os.path.join(models_dir, f'{category}_model.pkl')
    vectorizer_filename = os.path.join(models_dir, f'{category}_vectorizer.pkl')

     # Load the model and vectorizer
    loaded_model = joblib.load(model_filename)
    loaded_vectorizer = joblib.load(vectorizer_filename)

    X_sample = loaded_vectorizer.transform(df['questions'])

    predictions = loaded_model.predict(X_sample)
    probabilities = loaded_model.predict_proba(X_sample)

    df[f'prediction_{category}'] = predictions
    df[f'probability_0_{category}'] = probabilities[:, 0]
    df[f'probability_1_{category}'] = probabilities[:, 1]


# Function to calculate domain scores
def calculate_domain_score(row, category):
    """
    Calculates a score for each row based on model predictions and probabilities.
    """
    if row[f'prediction_{category}'] == 0:
        return (1 - row[f'prediction_{category}']) * row[f'probability_1_{category}']
    else:
        return row[f'probability_1_{category}']
    

# Main function to run the entire pipeline
def run_pipeline(predictions_filepath):
    # Load datasets
    wellbeing_df = pd.read_excel('posi_nega_w.xlsx')  # Wellbeing data
    functioning_df = pd.read_excel('posi_nega_f.xlsx')  # Functioning data

    # Define categories and corresponding dataframes
    categories = {
        'negative_wellbeing': wellbeing_df,
        'negative_functioning': functioning_df
    }

    # Train models and save them
    models, vectorizers = train_and_save_models(categories)

    # Filter sentences by domain using the provided predictions_filepath
    filtered_sentences_f_only, filtered_sentences_w_only, filtered_sentences_ps_only, filtered_sentences_rv_only = filter_sentences_by_domain(predictions_filepath)

     # Classify sentences for wellbeing and functioning
    classify_sentences(filtered_sentences_f_only, 'negative_functioning')
    classify_sentences(filtered_sentences_w_only, 'negative_wellbeing')

    # Calculate scores for each domain
    for category in categories.keys():
        if category == 'negative_functioning':
            filtered_sentences_f_only[f'{category}_score'] = filtered_sentences_f_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
        elif category == 'negative_wellbeing':
            filtered_sentences_w_only[f'{category}_score'] = filtered_sentences_w_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
        

    # Calculate overall scores
    overall_wellbeing_score = round(filtered_sentences_w_only['negative_wellbeing_score'].mean(), 4)
    overall_functioning_score = round(filtered_sentences_f_only['negative_functioning_score'].mean(), 4)
    overall_problemsandsymptoms_score = round(filtered_sentences_ps_only['probability_ps'].mean(), 4)
    # Filter sentences with probability > 0.5
    # high_wellbeing_sentences = filtered_sentences_w_only[filtered_sentences_w_only['negative_wellbeing_score'] > 0.5]
    # high_functioning_sentences = filtered_sentences_f_only[filtered_sentences_f_only['negative_functioning_score'] > 0.5]
    # high_problemsandsymptoms_sentences = filtered_sentences_ps_only[filtered_sentences_ps_only['probability_ps'] > 0.5]
    high_risk_sentences = filtered_sentences_rv_only[filtered_sentences_rv_only['probability_rv'] > 0.8] #for high risk category

    # Calculate the overall score
    # overall_wellbeing_score = high_wellbeing_sentences['negative_wellbeing_score'].sum() / len(filtered_sentences_w_only)
    # overall_functioning_score = high_functioning_sentences['negative_functioning_score'].sum() / len(filtered_sentences_f_only)
    # overall_problemsandsymptoms_score = high_problemsandsymptoms_sentences['probability_ps'].sum() / len(filtered_sentences_ps_only) 
    overall_risk_score = round(high_risk_sentences['probability_rv'].sum() / len(filtered_sentences_rv_only+filtered_sentences_f_only+filtered_sentences_ps_only+filtered_sentences_w_only),4) if not filtered_sentences_rv_only.empty else 0
    # overall_risk_score = 1 if not filtered_sentences_rv_only.empty else 0
    # overall_risk_score = filtered_sentences_rv_only['probability_rv'].mean() if not filtered_sentences_rv_only.empty else 0

    # Create a DataFrame to store overall scores
    overall_domainwise_scores_df = pd.DataFrame({
        'Domain': ['wellbeing', 'functioning', 'problemsandsymptoms', 'risk'],
        'Score': [overall_wellbeing_score, overall_functioning_score, overall_problemsandsymptoms_score, overall_risk_score]
    })

    # Return dataframes and scores
    return (
        filtered_sentences_w_only,  # Wellbeing results
        filtered_sentences_f_only,  # Functioning results
        filtered_sentences_ps_only,  # Problems & Symptoms results
        filtered_sentences_rv_only,  # Risk results
        overall_domainwise_scores_df, #domainwise scores
    )



def run_pipeline1(predictions_filepath):
    # Load datasets
    wellbeing_df = pd.read_excel('posi_nega_w.xlsx')  # Wellbeing data
    functioning_df = pd.read_excel('posi_nega_f.xlsx')  # Functioning data

    # Define categories and corresponding dataframes
    categories = {
        'negative_wellbeing': wellbeing_df,
        'negative_functioning': functioning_df
    }

    # Train models and save them
    models, vectorizers = train_and_save_models(categories)

    # Filter sentences by domain using the provided predictions_filepath
    filtered_sentences_f_only, filtered_sentences_w_only, filtered_sentences_ps_only, filtered_sentences_rv_only = filter_sentences_by_domain(predictions_filepath)

    # Classify sentences for wellbeing and functioning
    classify_sentences(filtered_sentences_f_only, 'negative_functioning')
    classify_sentences(filtered_sentences_w_only, 'negative_wellbeing')

    # Calculate scores for each domain
    for category in categories.keys():
        if category == 'negative_functioning':
            filtered_sentences_f_only[f'{category}_score'] = filtered_sentences_f_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
        elif category == 'negative_wellbeing':
            filtered_sentences_w_only[f'{category}_score'] = filtered_sentences_w_only.apply(lambda row: calculate_domain_score(row, category), axis=1)

    # Classify sentences for problems and symptoms and risk
    # Uncomment the following lines to handle additional categories if needed
    # for category in categories2.keys():
    #     if category == 'problems_and_symptoms':
    #         filtered_sentences_ps_only[f'{category}_score'] = filtered_sentences_ps_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
    #     elif category == 'risk':
    #         filtered_sentences_r_only[f'{category}_score'] = filtered_sentences_r_only.apply(lambda row: calculate_domain_score(row, category), axis=1)

    # Calculate overall scores
    overall_wellbeing_score = filtered_sentences_w_only['negative_wellbeing_score'].mean()
    overall_functioning_score = filtered_sentences_f_only['negative_functioning_score'].mean()
    overall_problemsandsymptoms_score = filtered_sentences_ps_only['probability_ps'].mean()  # Ensure this column exists
    overall_risk_score = filtered_sentences_rv_only['probability_rv'].mean()  # Ensure this column exists

    print(f"Overall wellbeing score: {overall_wellbeing_score}")
    print(f"Overall functioning score: {overall_functioning_score}")
    print(f"Overall problems and symptoms score: {overall_problemsandsymptoms_score}")
    print(f"Overall risk score: {overall_risk_score}")

    # Save the results
    filtered_sentences_w_only.to_excel('wellbeing_pn_results.xlsx', index=False)
    filtered_sentences_f_only.to_excel('functioning_pn_results.xlsx', index=False)
    filtered_sentences_ps_only.to_excel('ps_pn_results.xlsx', index=False)
    filtered_sentences_rv_only.to_excel('risk_pn_results.xlsx', index=False)

    print(f"Classification results for wellbeing saved to wellbeing_pn_results.xlsx")
    print(f"Classification results for functioning saved to functioning_pn_results.xlsx")
    print(f"Classification results for problems and symptoms saved to ps_pn_results.xlsx")
    print(f"Classification results for risk saved to risk_pn_results.xlsx")

    # return wellbeing_pn_results.xlsx, ps_pn_results.xlsx, risk_pn_results.xlsx

    # # Main function to run the entire pipeline
# def run_pipeline():
#     # Load datasets
#     wellbeing_df = pd.read_excel('posi_nega_w.xlsx')  # Wellbeing data
#     functioning_df = pd.read_excel('posi_nega_f.xlsx')  # Functioning data

#     # Define categories and corresponding dataframes
#     categories = {
#         'negative_wellbeing': wellbeing_df,
#         'negative_functioning': functioning_df
#     }

#     # categories2 = {
#     #     'problems_and_symptoms': ps,
#     #     'risk': r
#     # }
#     # Train models and save them
#     models, vectorizers = train_and_save_models(categories)

#     # Filter sentences by domain
#     predictions_filepath = 'output_predictions(3).xlsx'  # Path to the predictions file
#     filtered_sentences_f_only, filtered_sentences_w_only, filtered_sentences_ps_only, filtered_sentences_r_only = filter_sentences_by_domain(predictions_filepath)


#     # Classify sentences for wellbeing and functioning
#     classify_sentences(filtered_sentences_f_only, 'negative_functioning')
#     classify_sentences(filtered_sentences_w_only, 'negative_wellbeing')

#     # Calculate scores for each domain
#     for category in categories.keys():
#         if category == 'negative_functioning':
#             filtered_sentences_f_only[f'{category}_score'] = filtered_sentences_f_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
#         elif category == 'negative_wellbeing':
#             filtered_sentences_w_only[f'{category}_score'] = filtered_sentences_w_only.apply(lambda row: calculate_domain_score(row, category), axis=1)

#     # # Classify sentences for problems and symptoms and risk
#     # for category in categories.keys():
#     #     if category == 'problems_and_symptoms':
#     #         filtered_sentences_ps_only[f'{category}_score'] = filtered_sentences_ps_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
#     #     elif category == 'r':
#     #          filtered_sentences_r_only[f'{category}_score'] = filtered_sentences_r_only.apply(lambda row: calculate_domain_score(row, category), axis=1)
   
#     # Calculate overall scores
#     overall_wellbeing_score = filtered_sentences_w_only['negative_wellbeing_score'].mean()
#     overall_functioning_score = filtered_sentences_f_only['negative_functioning_score'].mean()
#     overall_problemsandsymptoms_score = filtered_sentences_ps_only['probability_ps'].mean()
#     overall_risk_score = filtered_sentences_r_only['probability_r'].mean()

#     print(f"Overall wellbeing score: {overall_wellbeing_score}")
#     print(f"Overall functioning score: {overall_functioning_score}")
#     print(f"Overall problems and symptoms score: {overall_problemsandsymptoms_score}")
#     print(f"Overall risk score: {overall_risk_score}")
#     # Save the results
#     filtered_sentences_w_only.to_excel('wellbeing_pn_results.xlsx', index=False)
#     filtered_sentences_f_only.to_excel('functioning_pn_results.xlsx', index=False)
#     filtered_sentences_ps_only.to_excel('ps_pn_results.xlsx', index=False)
#     filtered_sentences_r_only.to_excel('risk_pn_results.xlsx', index=False)

#     print(f"Classification results for wellbeing saved to wellbeing_pn_results.xlsx")
#     print(f"Classification results for functioning saved to functioning_pn_results.xlsx")
#     print(f"Classification results for ps saved to ps_pn_results.xlsx")
#     print(f"Classification results for risk saved to risk_pn_results.xlsx")
#     filtered_sentences_w_only.to_excel('wellbeing_pn_results.xlsx', index=False)
#     filtered_sentences_f_only.to_excel('functioning_pn_results.xlsx', index=False)
#     filtered_sentences_ps_only.to_excel('ps_pn_results.xlsx', index=False)
#     filtered_sentences_r_only.to_excel('risk_pn_results.xlsx', index=False)
# Run the pipeline
# if __name__ == "__main__":
#     run_pipeline()