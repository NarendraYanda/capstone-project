
# PDF Data Extraction and Preprocessing Pipeline

This repository contains a Python-based pipeline for extracting and preprocessing textual and tabular data from PDF files, combining Optical Character Recognition (OCR) and Natural Language Processing (NLP) techniques. The primary goal is to handle legal documents, focusing on extracting tables with their surrounding context and preprocessing text for further analysis.

## Features
- **PDF Text Extraction**: Extracts both text and tables from PDF files using `pdfplumber` and `tabula-py`.
- **OCR Integration**: Uses Tesseract OCR for pages where direct text extraction isn't possible.
- **Table Context Extraction**: Captures lines of text surrounding tables, ensuring that important context is preserved.
- **Text Preprocessing**: Performs text normalization, tokenization, stopword removal, lemmatization, and Named Entity Recognition (NER) using `nltk` and `spacy`.
- **Batch Processing**: Handles multiple PDF files from a specified directory, processing each one individually.

## Setup Instructions

### Step 1: Mount Google Drive
If using Google Colab, start by mounting your Google Drive to access files stored there.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Install Required Libraries
Install the necessary libraries to handle OCR, PDF extraction, and NLP tasks.

```bash
!apt-get install tesseract-ocr
!pip install pytesseract pdfplumber tabula-py
```

### Step 3: Import Libraries
Import the necessary Python libraries for handling files, PDFs, OCR, and NLP.

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
Ensure you have the necessary NLTK resources for tokenization, stopwords, and lemmatization.

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### Step 5: Load Spacy Model for Named Entity Recognition
Load the SpaCy model to perform Named Entity Recognition (NER) on the extracted text.

```python
nlp = spacy.load('en_core_web_sm')
```

### Step 6: Text Normalization
Standardize the text by converting it to lowercase, removing punctuation, and handling other text-cleaning tasks.

```python
def text_normalization(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### Step 7: Table and Context Extraction
Extract tables from the PDF and capture the surrounding text for contextual understanding.

```python
def extract_tables_and_context(pdf_path, page_text, page_num, previous_page_text=None, lines_above=3, lines_below=3):
    tables = tabula.read_pdf(pdf_path, pages=page_num, multiple_tables=True)
    ...
```

### Step 8: PDF Content Extraction with OCR
Extract text from each page of the PDF, falling back on OCR for pages without direct text extraction.

```python
def extract_pdf_content_with_ocr(pdf_path, lines_above=3, lines_below=3):
    ...
```

### Step 9: Process Multiple Files
Process all PDF files in a specified directory, extracting content and applying the preprocessing pipeline.

```python
def process_files(pdf_directory):
    ...
```

### Step 10: Run the Pipeline
Specify the directory containing your PDF files and process each file for text and table extraction.

```python
pdf_directory = "/content/drive/MyDrive/AIML/Capstone-Project/data/LimitedData/PDF"
all_preprocessed_data = process_files(pdf_directory)
```

### Step 11: Review Output
Print out the preprocessed text, tables, and extracted context for each file processed.

```python
for data in all_preprocessed_data:
    print(f"\nFile: {data['file_name']}")
    print(f"Normalized Text:\n{data['normalized_text'][:500]}")
    ...
```

## Output
For each processed PDF file, the pipeline generates:
- **Normalized Text**: Text from the PDF, cleaned and preprocessed.
- **Lemmatized Words**: Processed tokens after lemmatization.
- **Named Entities**: Recognized named entities such as names, locations, and dates.
- **Tables with Context**: Extracted tables from the PDF along with the text immediately above and below the table for added context.


## Libraries Used
- **Tesseract-OCR**: For Optical Character Recognition (OCR).
- **Pytesseract**: Python wrapper for Tesseract.
- **pdfplumber**: To extract text and metadata from PDFs.
- **tabula-py**: Extract tables from PDF documents.
- **NLTK**: Tokenization, stopword removal, and lemmatization.
- **SpaCy**: Named Entity Recognition (NER) using pre-trained models.
