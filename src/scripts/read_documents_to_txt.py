import os
import pickle
import pdfplumber
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv
import re
from docx import Document
from spellchecker import SpellChecker
import wordninja

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Download required NLTK data
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stopwords
nltk.download('wordnet')  # Lemmatizer

# Initialize spellchecker
spell = SpellChecker()

# Suppress specific warnings (e.g., matplotlib warnings)
import warnings
warnings.filterwarnings("ignore", message="usetex mode requires TeX.")

# Define global constants for file paths
DIRECTORY_PATH = r'C:\Users\Aspire5 15 i7 4G2050\Project_2\AviationRAG\data\documents'
TEXT_OUTPUT_DIR = r'C:\Users\Aspire5 15 i7 4G2050\Project_2\AviationRAG\data\ProcessedTex'
TEXT_EXPANDED_DIR = r'C:\Users\Aspire5 15 i7 4G2050\Project_2\AviationRAG\data\ProcessedTextExapanded'
PKL_FILENAME = r'C:\Users\Aspire5 15 i7 4G2050\Project_2\AviationRAG\data\raw\aviation_corpus.pkl'

# Global stopwords
STOP_WORDS = set(stopwords.words('english'))

# Functions to support the main document reading and processing routine
def load_abbreviation_dict():
    abbreviation_dict = {}
    try:
        with open('abbreviations.csv', mode='r') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if len(rows) < 2:
                    continue
                abbreviation_dict[rows[0].strip()] = rows[1].strip()
    except FileNotFoundError:
        print("Error: The file 'abbreviations.csv' was not found.")
    except Exception as e:
        print(f"An error occurred while loading the abbreviation dictionary: {e}")
    return abbreviation_dict

def split_connected_words_improved(text):
    words = re.findall(r'\w+|\W+', text)
    split_words = []
    for word in words:
        if len(word) > 15 and word.isalnum():
            split_parts = re.findall('[A-Z][a-z]*|[a-z]+|[0-9]+', word)
            split_words.extend(split_parts)
        else:
            split_words.append(word)
    # Use wordninja to further split any remaining connected words
    split_words = ' '.join(split_words)
    split_words = ' '.join(wordninja.split(split_words))
    return split_words

def filter_non_sense_strings(text):
    words = text.split()
    cleaned_words = []
    for word in words:
        if re.match(r'^[a-zA-Z]+$', word) and len(set(word.lower())) > 3:
            # Keep words with a good mix of characters and only alphabetic
            cleaned_words.append(word)
    return ' '.join(cleaned_words)

def preprocess_text_with_sentences(text):
    # Tokenize into sentences
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        # Clean and process each sentence
        cleaned_sentence = ' '.join(
            token.lemma_.lower() for token in sent
            if token.is_alpha and token.text.lower() not in STOP_WORDS
        )
        if cleaned_sentence:  # Avoid empty sentences
            sentences.append(cleaned_sentence)
    return ' '.join(sentences)  # Return as a single string with sentence boundaries

def extract_personal_names(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

def extract_entities_and_pos_tags(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return entities, pos_tags

def expand_abbreviations_in_text(text, abbreviation_dict):
    words = text.split()
    expanded_words = []
    for word in words:
        if word.lower() in abbreviation_dict:
            expanded_words.append(abbreviation_dict[word.lower()])
        else:
            expanded_words.append(word)
    return ' '.join(expanded_words)

def extract_text_from_pdf_with_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join([page.extract_text() + '\n' for page in pdf.pages])
            return text
    except Exception as e:
        print(f"Failed to process PDF {pdf_path}: {e}")
        return ""
def read_documents_from_directory(directory_path, text_output_dir=None, text_expanded_dir=None, existing_documents=None):
    if existing_documents is None:
        existing_documents = []
    
    os.makedirs(text_expanded_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)


    existing_files = {doc['filename'] for doc in existing_documents}
    new_documents = []
    abbreviation_dict = load_abbreviation_dict()  # Load abbreviations once
    lemmatizer = WordNetLemmatizer()

    for filename in os.listdir(directory_path):
        if filename in existing_files:
            continue

        file_path = os.path.join(directory_path, filename)
        text = ''
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf_with_pdfplumber(file_path)
        elif filename.endswith(".docx"):
            try:
                doc = Document(file_path)
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except Exception as e:
                print(f"Failed to process DOCX {filename}: {e}")
                continue
        else:
            continue  # Skip unsupported files

        if not text:
            continue

        # Expand abbreviations in the raw text
        expanded_text = expand_abbreviations_in_text(text, abbreviation_dict)
        raw_text = expanded_text

        # Split connected words and expand abbreviations in the preprocessed text
        expanded_text = split_connected_words_improved(expanded_text)

        # Filter non-sense strings
        expanded_text = filter_non_sense_strings(expanded_text)

        # Preprocess the extracted text
        preprocessed_text = preprocess_text_with_sentences(expanded_text)

        # Extract personal names
        personal_names = extract_personal_names(preprocessed_text)

        # Extract entities and POS tags
        entities, pos_tags = extract_entities_and_pos_tags(preprocessed_text)

        # Tokenization and text cleaning
        tokens = word_tokenize(preprocessed_text)
        cleaned_tokens = [token.lower() for token in tokens if token.isalpha() and len(token) > 2]
        tokens_without_stopwords = [token for token in cleaned_tokens if token not in STOP_WORDS]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_without_stopwords]

        # Save the expanded_text file if directory is specified
        if text_expanded_dir:
            output_file_path = os.path.join(text_expanded_dir, f'{filename}.txt')
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(raw_text)
            print(f"Expanded text saved to: {output_file_path}")
               
        # Save as plain text file if output directory is specified
        if text_output_dir:
            output_file_path = os.path.join(text_output_dir, f'{filename}.txt')
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(preprocessed_text)
            print(f"Text saved to: {output_file_path}")

        # Store the text and tokens in a simplified format
        new_documents.append({
            'filename': filename,
            'text': preprocessed_text,
            'tokens': lemmatized_tokens,
            'personal_names': personal_names,
            'entities': entities,
            'pos_tags': pos_tags
        })

    return existing_documents + new_documents

# Main routine
def main():
    documents = None
    if os.path.exists(PKL_FILENAME):
        with open(PKL_FILENAME, 'rb') as file:
            documents = pickle.load(file)

    if documents is None:
        print("Reading documents from directory...")
        documents = read_documents_from_directory(DIRECTORY_PATH, TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR)
    else:
        print("Appending new documents to the existing list...")
        documents = read_documents_from_directory(DIRECTORY_PATH, TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR, documents)

    # Save the updated list
    with open(PKL_FILENAME, 'wb') as file:
        pickle.dump(documents, file)

    print(f"Total documents: {len(documents)}")

if __name__ == '__main__':
    main()