
# PDF and Table Extraction with OCR and NLP Pipeline


## Features
- **Text and Table Extraction**: Extracts text from PDFs using `pdfplumber` and tables using `tabula-py`.
- **OCR Integration**: Pages that cannot be extracted directly are processed using Tesseract OCR (`pytesseract`).
- **Contextual Table Extraction**: Extracts tables from PDFs along with lines of text before and after the table for contextual understanding.
- **Text Preprocessing**: Includes steps like text normalization, stopword removal, lemmatization, and Named Entity Recognition (NER) using `spacy`.
- **Batch Processing**: Handles multiple PDF files at once from a specified directory.

## Setup Instructions

### Step 1: Mount Google Drive ( Colab )

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Install Required Libraries
Install the necessary packages for text extraction, OCR, and NLP tasks.

```bash
!apt-get install tesseract-ocr
!pip install pytesseract pdfplumber tabula-py
```

### Step 3: Import Libraries
Import the necessary libraries for text extraction, OCR, and Natural Language Processing (NLP).

```python
import os
import re
import tabula
import pdfplumber
import pytesseract
from PIL import Image
import io
import pandas as pd
import spacy
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
```

### Step 4: Download NLTK Data
Download the required resources for NLTK functions, such as tokenization, stopword removal, and lemmatization.

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

---

## Functions Overview

### Text Normalization Function
```python
def text_normalization(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text is re.sub(r'\s+', ' ', text).strip()
    return text
```
- **Purpose**: This function cleans the text by converting it to lowercase, removing punctuation and numbers, and normalizing whitespace.
- **Output**: The cleaned and normalized text, which is easier to process during NLP tasks.

---

### Lemmatization Function
```python
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(tokens)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tagged]
    return lemmatized_words
```
- **Purpose**: This function lemmatizes tokens, reducing words to their base form based on their part of speech (e.g., running → run).
- **Output**: A list of lemmatized words, which helps to reduce the dimensionality of the vocabulary.

---

### POS Tagging and Lemmatization Helper Function
```python
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```
- **Purpose**: This function maps the POS tags used by NLTK’s `pos_tag` function to WordNet’s format, which is necessary for accurate lemmatization.
- **Output**: Returns the appropriate part of speech tag for each word.

---

### Named Entity Recognition (NER) Function
```python
def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities
```
- **Purpose**: This function identifies and extracts named entities (e.g., people, organizations, dates) from the text using the SpaCy NER model.
- **Output**: A list of named entities and their corresponding types (e.g., person, organization).

---

### Stopword Removal Function
```python
def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words
```
- **Purpose**: This function removes common English stopwords (e.g., "the", "is", "in"), which don’t contribute much meaning in NLP tasks.
- **Output**: A list of words with stopwords removed, focusing only on meaningful words.

---

### Table Extraction with Context Function
```python
def extract_tables_and_context(pdf_path, page_text, page_num, previous_page_text=None, lines_above=3, lines_below=3):
    tables = tabula.read_pdf(pdf_path, pages=page_num, multiple_tables=True)
    ...
```
- **Purpose**: This function extracts tables from a PDF page using `tabula-py` and captures lines of text around the table for added context.
- **Output**: A list of extracted tables and their surrounding textual context.

---

### OCR and Table Extraction Function
```python
def extract_pdf_content_with_ocr(pdf_path, lines_above=3, lines_below=3):
    ...
```
- **Purpose**: This function extracts content from PDF files. If text extraction fails, it uses Tesseract OCR to extract text from images.
- **Output**: Extracted text and tables along with context, for each page of the PDF.

---

### Batch Processing Function
```python
def process_files(pdf_directory):
    ...
```
- **Purpose**: This function processes multiple PDF files in a given directory, applying text extraction, OCR, and preprocessing steps like tokenization, lemmatization, and table extraction.
- **Output**: A list containing pre-processed data for each PDF file.

---

## Running the Pipeline

### Step 1: Run the Pipeline
Specify the directory containing your PDF files and run the `process_files` function to extract and preprocess the data.

```python
pdf_directory = "/content/drive/MyDrive/AIML/Capstone-Project/data/LimitedData/PDF"
all_preprocessed_data = process_files(pdf_directory)
```

### Step 2: Review the Output
Print the processed data, including normalized text, lemmatized words, named entities, and tables (with surrounding context).

```python
for data in all_preprocessed_data:
    print(f"\nFile: {data['file_name']}")
    print(f"Normalized Text:\n{data['normalized_text'][:500]}")
    ...
```

---

## Overall Collective Output

For each processed PDF, the output includes:
1. **Normalized Text**: The cleaned, lowercased, and punctuation-free text from the PDF.
2. **Lemmatized Words**: Words reduced to their base forms (e.g., "running" → "run").
3. **Named Entities**: Detected entities like people, organizations, and dates.
4. **Tables with Context**: Extracted tables, along with text lines that appear before and after each table.

This combined output provides both structured (tables) and unstructured (text) information, allowing for comprehensive analysis of the PDF content.

---

## Libraries Used
- **Tesseract-OCR**: For Optical Character Recognition (OCR).
- **Pytesseract**: Python wrapper for Tesseract.
- **pdfplumber**: To extract text and metadata from PDFs.
- **tabula-py**: Extract tables from PDF documents.
- **NLTK**: Tokenization, stopword removal, and lemmatization.
- **SpaCy**: Named Entity Recognition (NER).

---

