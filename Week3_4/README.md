
# Automated Legal Document Summarization - Checkpoint 2 Solution

## Overview

This project uses **FAISS** for document retrieval, **K-Means clustering** to group similar documents, and dynamic summarization techniques (both extractive and abstractive) using **BERT** and **LLaMA 3.1**. The system provides query-based and full-document summarization capabilities for legal documents.

## Key Components

- **Preprocessing**: Cleans text, performs tokenization, lemmatization, named entity recognition (NER), and table extraction from PDFs.
- **FAISS**: Efficient vector search for document embeddings, enabling fast query-based retrieval.
- **K-Means Clustering**: Groups documents into clusters to limit query searches to relevant documents.
- **BERT**: Used for generating document embeddings and extractive summarization.
- **LLaMA 3.1**: Provides abstractive summarization to generate human-like summaries.

## Functions & Steps

1. **Preprocessing**:
   - `process_files(pdf_directory)`: Processes PDF files by normalizing text, tokenizing, extracting tables, and performing NER.

2. **FAISS Setup**:
   - `initialize_faiss_index(embedding_dimension)`: Initializes a FAISS index for storing document embeddings.
   - `add_embeddings_to_faiss(index, embeddings)`: Adds BERT-generated embeddings to the FAISS index.

3. **Clustering**:
   - `perform_kmeans_clustering(embeddings, num_clusters)`: Performs K-Means clustering to group similar documents into clusters.

4. **Summarization**:
   - `bert_extractive_summary(text, named_entities)`: Generates an extractive summary by ranking sentences using BERT embeddings.
   - `generate_llama_summary(text)`: Uses LLaMA 3.1 to generate an abstractive summary of the document.

5. **Dynamic Summary Mode**:
   - `dynamic_summary_mode(...)`: Handles query-based and full-document summarization. Searches for relevant documents using FAISS within specific clusters.

## Models & Libraries Used

- **BERT**: For embedding generation and extractive summarization.
- **FAISS**: Efficient similarity search for document retrieval.
- **K-Means**: Clustering for grouping documents.
- **LLaMA 3.1**: Abstractive summarization to provide human-readable summaries.
- **Spacy**: For named entity recognition (NER).
- **Tabula & PDFPlumber**: Extracts tables and text from PDFs.

## Solution Achieved

- **Efficient Query Handling**: FAISS and K-Means are used to retrieve documents relevant to the user’s query, limiting searches to relevant clusters.
- **Dynamic Summarization**: Supports both query-based and full-document summarization using BERT and LLaMA.
- **Optimized Performance**: Reuses document embeddings and clusters to minimize redundant computations.

## How to Use

1. **Preprocess Documents**:
   ```python
   all_preprocessed_data = process_files("path_to_pdf_folder")
   ```

2. **Prepare FAISS and K-Means Clusters**:
   ```python
   faiss_index, all_document_embeddings, clusters, kmeans_model = prepare_faiss_and_clusters(all_preprocessed_data, num_clusters=5)
   ```

3. **Query-Based Summarization**:
   ```python
   dynamic_summary_mode(all_preprocessed_data, faiss_index, clusters, kmeans_model, mode="query", query="intellectual property law", top_k=5)
   ```

4. **Full-Document Summarization**:
   ```python
   dynamic_summary_mode(all_preprocessed_data, faiss_index, clusters, kmeans_model, mode="full")
   ```

This solution enables fast, efficient summarization of legal documents by combining clustering, vector search, and deep learning models.
