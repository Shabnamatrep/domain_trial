####For text sentences extraction from the story pdf

from pydantic import BaseModel
from pathlib import Path
from typing import List
import pandas as pd
import re
import os
import pdfplumber
 # Make sure to install nltk and download the 'punkt' package
import nltk
from nltk.tokenize import sent_tokenize

# # # Set NLTK data path explicitly for deployment
# os.environ["NLTK_DATA"] = "nltk_data"
# # Ensure you have the punkt tokenizer downloaded
# download_dir = os.path.abspath('nltk_data')
# # nltk.download('punkt')
# nltk.download('punkt', download_dir=download_dir)
# Set the NLTK data path to use the pre-downloaded folder
# os.environ["NLTK_DATA"] = "nltk_data"

# Verify that 'punkt' is available (this won't download it again)

# Specify the path where your NLTK data is located
nltk_data_path = "nltk_data"  # Update this path
nltk.data.path.append(nltk_data_path)
nltk.data.find('tokenizers/punkt')

# import ntlk.data
# root = os.path.dirname(path.abspath(__file__))
# download_dir = os.path.join(root, 'my_nltk_dir')
# nltk.data.load(
#     os.path.join(download_dir, 'tokenizers/punkt/english.pickle')
# )

class Text:
    def __init__(self, text: str, text_id: str):
        self.text = text
        self.text_id = text_id

    @classmethod
    def from_file(cls, file_path: Path) -> 'Text':
        """Read a PDF file and extract its text content."""
        if file_path.suffix == ".pdf":
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
                return None
        else:
            print(f"Unsupported file format for text extraction: {file_path.suffix}")
            return None
        return cls(text=text, text_id=file_path.name)


def clean_sentences(text: str) -> List[str]:
    cleaned_sentences = []

    # Helper function to check if a sentence is meaningful
    def is_meaningful(sentence: str) -> bool:
        if not isinstance(sentence, str):  # Ensure it's a string
            return False
        if sentence.strip().isdigit():  # Remove rows with only numbers
            return False
        if any(char.isdigit() for char in sentence):  # Remove rows with too many numbers
            return False
        return len(sentence.split()) > 2  # Keep sentences with more than 2 words

    # Pattern to match Roman numerals (both lowercase and uppercase)
    roman_numeral_pattern = r'\b[Mm]{0,1}(CM|cm|CD|cd|D|d)?(C{0,3}|c{0,3})(XC|xc|XL|xl|L|l)?(X{0,3}|x{0,3})(IX|ix|IV|iv|V|v|I{0,3}|i{0,3})\b'

    # List of conjunctions to check for
    conjunctions = ['and', 'but', 'or', 'nor', 'for', 'so', 'yet']

    # Split the text into sentences using NLTK
    sentences = sent_tokenize(text)
    
    # Remove the first sentence (optional)
    if len(sentences) > 2:
        sentences = sentences[2:]
    
    for sentence in sentences:
        # Normalize whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        # Remove headers, footers, unwanted strings, and Roman numerals
        sentence = re.sub(r'(?i)(OceanofPDF\.com|Page \d+|Title Page|Dedication|Contents|Epigraph)', '', sentence)
        sentence = re.sub(roman_numeral_pattern, '', sentence)  # Remove Roman numerals

        # Remove special characters except some punctuation (optional)
        sentence = re.sub(r'[^a-zA-Z0-9.,?!\s]', '', sentence)

        # Convert to lowercase (optional)
        sentence = sentence.lower()
        
        # Strip trailing periods or punctuation-only sentences
        sentence = sentence.rstrip('.')

        # Apply the is_meaningful filter
        if not is_meaningful(sentence):
            continue

        # Remove standalone words that might be meaningless (e.g., single words)
        if len(sentence.split()) == 1 and re.fullmatch(r'\b\w+\.?com\b', sentence):
            continue

        # Check if the sentence starts with a conjunction
        first_word = sentence.split()[0]
        if first_word in conjunctions:
            # Remove the conjunction
            sentence = ' '.join(sentence.split()[1:]).strip()

        # Add to cleaned sentences if it passes all checks
        cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def process_pdf_and_return_dataframe(pdf_path: Path) -> pd.DataFrame:
    allowed_types = (".pdf",)
    files = [file for file in pdf_path.glob("*") if file.suffix in allowed_types]
    out = []

    for file in files:
        print(f"Processing {file}")
        text_data = Text.from_file(file)
        if text_data:
            cleaned_sentences = clean_sentences(text_data.text)
            out.append(pd.DataFrame({"questions": cleaned_sentences, "File Name": text_data.text_id}))

    if out:
        df = pd.concat(out, ignore_index=True)
        return df  # Return the DataFrame containing the cleaned sentences
    else:
        print("No data to process. Please check the input files.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found