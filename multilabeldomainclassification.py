##############************************************Multi-Label Domain Classification System***************************************
#Pretrained DistilBert models are trained using labelled domain classification data and predictions are made for the story sentences are saved
#as output_predictions with multilabel predictions and their respective probabilities

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
import os
from dotenv import load_dotenv
from extract_sentences import *
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax

#F:\Rep_Well_Being\Projects_Responsible\Domains\multilabeldomain_dbert_trainingnsaving.py
#if training for the first time uncomment the following 3 lines below
from multilabeldomain_dbert_trainingnsaving import run_all
# if __name__ == "__main__":
run_all()


# Load environment variables from .env file
load_dotenv()
# Set the Hugging Face token. 
hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')

#print(hugging_face_token)


# Define the binary classifier model for domain classification
class DistilBertBinaryClassifier(nn.Module):
    def __init__(self):
        super(DistilBertBinaryClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]  # CLS token output
        logits = self.classifier(cls_output)
        return logits

# Define the sequence classifier for risk victim classification
class DistilBertSequenceClassifier(nn.Module):
    def __init__(self):
        super(DistilBertSequenceClassifier, self).__init__()
        # Initialize DistilBertForSequenceClassification with the number of labels
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    def forward(self, input_ids, attention_mask):
        # Forward pass through the model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Return the logits for classification
    
# Inference function for a domain model
def infer(sentence, model, tokenizer):
    # Set the model to evaluation mode
    # model = DistilBertBinaryClassifier()
    # model.load_state_dict(torch.load('domain_models\distilbertbinary_r\distilbert_binary_classifier_r.pth'))
    model.eval()

    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Perform inference
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = sigmoid(logits)
        predicted_label = (probs > 0.5).float()  # Convert to binary prediction (0 or 1)

    return predicted_label.item(), probs.item()  # Return label and probability


#Defining inference for the risk victim classifier
def infer_rvnv(sentence, model, tokenizer):
    # Tokenize the input sentence
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Perform inference
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = softmax(logits, dim=1)  # Apply softmax to get probabilities
        v_probs = probs[0][1].item() # Get probabilities
        if v_probs > 0.8:
            plabel = 1
        else:
            plabel = 0
        #plabel = (probs > 0.8).float()
        #plabel = torch.argmax(probs, dim=1).item()  # Get the predicted label (0 or 1)

    return plabel, v_probs # Return label and corresponding probability  # Return label and corresponding probability

def process_excel(input_file_df, model_names):
    # Load the Excel file
    df = input_file_df

    # Check if the 'questions' column exists
    if 'questions' not in df.columns:
        raise ValueError("The input DataFrame must contain a 'questions' column.")
    
     # Define the base directory where models are stored (can be dynamically set)
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script

    # Initialize lists to store predictions and probabilities for each model
    for model_name in model_names:

        # try:
        #     model.load_state_dict(torch.load('path_to_model.pth'))
        # except Exception as e:
        #     print(f"Error loading model: {e}")
        #     raise
        # Construct the model file path
        model_path = os.path.join(base_dir, 'multilabel_domain_model', f'distilbert_binary_classifier_{model_name}.pth')
        model = DistilBertBinaryClassifier()
        try:
            if model_path:

                # model.load_state_dict(torch.load(model_path))
                # state_dict = torch.load(rf'multilabel_domain_model\distilbert_binary_classifier_{model_name}.pth')
                # model.load_state_dict(state_dict, strict=False)
                # model.load_state_dict(torch.load(rf'multilabel_domain_model\distilbert_binary_classifier_{model_name}.pth', weights_only=True))
                
                # Construct the tokenizer path
                tokenizer_path = os.path.join(base_dir, 'multilabel_domain_model', f'distilbert_binary_tokenizer_{model_name}')
                
                # Load the tokenizer
                tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path, token=hugging_face_token)

            else:
                print(f"Model file or tokenizer file not found for {model_name}")
                model.load_state_dict(torch.load(rf'multilabel_domain_model\distilbert_binary_classifier_{model_name}.pth'))
                tokenizer = DistilBertTokenizer.from_pretrained(
                rf'multilabel_domain_model\distilbert_binary_tokenizer_{model_name}/',
                token=hugging_face_token
                )
                
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise
            
                
                # Get the domain predictions for each question
        # # Construct the tokenizer path
        # tokenizer = DistilBertTokenizer.from_pretrained(
        #     rf'multilabel_domain_model\distilbert_binary_tokenizer_{model_name}/',
        #     token=hugging_face_token
        # )
        
        predictions = []
        probabilities = []

        # Loop through each question
        for question in df['questions']:
            predicted_label, probability = infer(question, model, tokenizer)
            predictions.append(model_name if predicted_label == 1 else 0)
            probabilities.append(round(probability, 4))

        # Add predictions and probabilities to DataFrame
        df[f'predicted_label_{model_name}'] = predictions
        df[f'probability_{model_name}'] = probabilities

        # If the model name is 'r', process further for sentences with probability > 0.5
        if model_name == 'r':
            # Filter sentences with probability greater than 0.5 for model 'r'
            filtered_df = df[df[f'probability_{model_name}'] > 0.5].copy()

            # Prepare to run the second model
            model_rv=DistilBertSequenceClassifier()
            # model_rv.load_state_dict(torch.load('distilbert_seq_classifier_rv.pth', weights_only=True))
            # # model_rv = DistilBertForSequenceClassification.from_pretrained('distilbert_seq_classifier_rv.pth')
            # tokenizer_rv = DistilBertTokenizer.from_pretrained('tokenizer_rv',token=hugging_face_token)
            try:
                if base_dir:
                    model_rv.load_state_dict(torch.load(os.path.join(base_dir, 'distilbert_seq_classifier_rv.pth')))
                    
                    # Load the tokenizer for the second model
                    tokenizer_rv = DistilBertTokenizer.from_pretrained(os.path.join(base_dir, 'tokenizer_rv'), token=hugging_face_token)
                else:
                    print(f"Model file rv or tokenizer file not found")
                    model_rv.load_state_dict(torch.load('distilbert_seq_classifier_rv.pth'))
                    tokenizer_rv = DistilBertTokenizer.from_pretrained('tokenizer_rv', token=hugging_face_token)
            except Exception as e:
                print(f"Error loading model rv or tokenizer: {e}")
                raise

            rv_predictions = []
            rv_probabilities = []

            # Loop through the filtered questions
            for question in filtered_df['questions']:
                predicted_label, probability = infer_rvnv(question, model_rv, tokenizer_rv)
                rv_predictions.append('rv' if predicted_label == 1 else 0)  # Label as 'rvnv' for 1
                rv_probabilities.append(round(probability, 4))

            # Add predictions and probabilities for the second model
            filtered_df['predicted_label_rv'] = rv_predictions
            filtered_df['probability_rv'] = rv_probabilities
            # Merge the filtered results back into the original DataFrame
            df = pd.merge(
                df,
                filtered_df[['questions', 'predicted_label_rv', 'probability_rv']],
                on='questions',
                how='left'
            )
            
            # Return both original and final DataFrames
            # return df, filtered_df

    # Return the original DataFrame if no filtering was applied
    return df

# Function to process a CSV file with multiple models
def get_new_filename(output_file):
    if not os.path.exists(output_file):
        return output_file
    
    base, ext = os.path.splitext(output_file)
    counter = 1
    new_output_file = f"{base}({counter}){ext}"
    
    while os.path.exists(new_output_file):
        counter += 1
        new_output_file = f"{base}({counter}){ext}"
    
    return new_output_file

# def process_excel2(input_file_df, model_names):
#     # Load the Excel file
#     # df = pd.read_excel(input_file)
#     df = input_file_df
#     # Check if the 'questions' column exists
#     if 'questions' not in df.columns:
#         raise ValueError("The input Excel file must contain a 'questions' column.")

#     # Initialize lists to store predictions and probabilities for each model
#     for model_name in model_names:
#         model = DistilBertBinaryClassifier()
#         model.load_state_dict(torch.load(rf'multilabel_domain_model\distilbert_binary_classifier_{model_name}.pth'))
#         tokenizer = DistilBertTokenizer.from_pretrained(
#             rf'multilabel_domain_model\distilbert_binary_tokenizer_{model_name}/',
#             token=hugging_face_token
#         )
        
#         predictions = []
#         probabilities = []

#         # Loop through each question
#         for question in df['questions']:
#             predicted_label, probability = infer(question, model, tokenizer)
#             predictions.append(model_name if predicted_label == 1 else 0)
#             probabilities.append(round(probability, 4))

#         # Add predictions and probabilities to DataFrame
#         df[f'predicted_label_{model_name}'] = predictions
#         df[f'probability_{model_name}'] = probabilities


#     # Return DataFrame instead of saving to a file
#     return df

# # Function to process an Excel file with multiple models
# def process_excel1(input_file, output_file, model_names):
#     # Load the Excel file
#     df = pd.read_excel(input_file)

#     # Check if the 'sentence' column exists
#     if 'questions' not in df.columns:
#         raise ValueError("The input Excel file must contain a 'questions' column.")

#     # Initialize lists to store predictions and probabilities for each model
#     for model_name in model_names:
#         model = DistilBertBinaryClassifier()
#         #if running for the first time
#         model.load_state_dict(torch.load(rf'multilabel_domain_model\distilbert_binary_classifier_{model_name}.pth'))  # Load the model weights

#         #model.load_state_dict(torch.load('domain_models\distilbertbinary_r\distilbert_binary_classifier_r.pth'))
#         # Load the tokenizer and provide the Hugging Face token
#         tokenizer = DistilBertTokenizer.from_pretrained(rf'multilabel_domain_model\distilbert_binary_tokenizer_{model_name}/',token=hugging_face_token)

        
#         # Initialize lists for storing results
#         predictions = []
#         probabilities = []

#         # Loop through each sentence in the column
#         for questions in df['questions']:
#             predicted_label, probability = infer(questions, model, tokenizer)
#             predictions.append(predicted_label)
#             probabilities.append(round(probability,4)) #round the probability to 4 decimal places

#         # Add predictions and probabilities to the DataFrame
#         #df[f'predicted_label_{model_name}'] = predictions
#         df[f'predicted_label_{model_name}'] = [model_name if label == 1 else label for label in predictions]
#         df[f'probability_{model_name}'] = probabilities

#         # Filter out rows where the predicted label is 0 (no prediction)
#         #df = df[df[f'predicted_label_{model_name}'] != 0]

#     # Save the updated DataFrame to a new Excel file
#     # df.to_excel(output_file, index=False)
#     # Save the updated DataFrame to a new Excel file with alignment
#     with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False)

#         # Access the workbook and worksheet
#         workbook = writer.book
#         worksheet = writer.sheets['Sheet1']  # Adjust 'Sheet1' if the sheet name is different

#         # Create a format for center alignment
#         center_align_format = workbook.add_format({'align': 'center'})

#         # Apply the center alignment format to specific columns (for predictions and probabilities)
#         for col_num, col_name in enumerate(df.columns):
#             if 'predicted_label_' in col_name or 'probability_' in col_name or 'domain' in col_name:
#                 worksheet.set_column(col_num, col_num, None, center_align_format)
#         print(f"Predictions saved to {output_file}")
#         return output_file


# Example call to process the PDF
# pdf_file_path = Path("Darkness_Visible__A_Memoir_of_Madness_-_William_Styron (1).pdf")  # Path to your PDF file
# output_excel_path = Path("cleaned_sentences_output.xlsx")  # Path to save the Excel file

# # Call the function and get the output Excel file path
# input_file = process_pdf_and_save(pdf_file_path, output_excel_path)

# # Main Block
# output_file = get_new_filename('output_predictions.xlsx')  # The output Excel file
# model_names = ['f', 'w', 'ps', 'r']  # List of model names

# # Now input_file is the Excel file generated by process_pdf_and_save
# process_excel(input_file, output_file, model_names)

# # Example call to process the PDF (you can call this function from other modules)
# pdf_file_path = Path("Darkness_Visible__A_Memoir_of_Madness_-_William_Styron (1).pdf")  # Path to your PDF file
# output_excel_path = Path("cleaned_sentences_output.xlsx")  # Path to save the Excel file
# process_pdf_and_save(pdf_file_path, output_excel_path)
# # Main Block:
# input_file = 'extracted_sentences.xlsx'  # The Excel file containing sentences
# #output_file = 'output_predictions.xlsx'  # The output Excel file
# output_file = get_new_filename('output_predictions.xlsx')
# model_names = ['f', 'w', 'ps', 'r']  # List of model names

# process_excel(input_file, output_file, model_names)
