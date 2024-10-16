
# Legal Document Processing and Query-based Summarization

This project aims to extract and summarize relevant information from legal documents using NLP, embeddings, FAISS indexing, and K-Means clustering. The solution supports both full-document and query-based summarization.

---

## Process Flow Overview

The system begins by reading and preprocessing legal documents, chunking them into meaningful sections, generating embeddings, clustering similar chunks, and providing query-based or full-document summaries.

---

### 1. **process_files()**
- **Purpose**: Load and preprocess all PDF files from a specified directory.
- **Model Used**: `ThreadPoolExecutor` to process multiple PDFs in parallel.
- **Why this approach?**  
    - Parallel processing improves speed by leveraging concurrency. 
    - **Alternatives**: Asynchronous I/O, though ThreadPoolExecutor provides simplicity.

**Improvement Suggestion**: If dealing with a large number of files, `asyncio` with non-blocking I/O could be more efficient.

---

### 2. **process_file()**
- **Purpose**: Extract text from individual PDFs and perform chunking and NER.
- **Model Used**: `pdfplumber` for PDF extraction and `spaCy` for Named Entity Recognition (NER).

- **Why NER?**  
    - Identifies crucial named entities (e.g., dates, parties) relevant for summarization.
    - **Alternatives**: `Hugging Face NER models` could provide better customization but require more resources.

**Improvement Suggestion**: Use more domain-specific NER models for legal texts to enhance accuracy.

---

### 3. **text_normalization()**
- **Purpose**: Clean and normalize text by removing punctuation and extra spaces.
- **Why Normalization?**  
    - Normalized text ensures consistent embeddings and similarity matching.
    - **Alternatives**: Lemmatization or stemming could be integrated for further improvements.

---

### 4. **nlp_based_chunking()**
- **Purpose**: Split text into chunks and categorize them into predefined legal categories.
- **Why Chunking?**  
    - Improves embedding quality by focusing on smaller, context-relevant pieces.
    - **Alternatives**: Sliding window techniques or adaptive chunking based on sentence structure could be explored.

---

### 5. **get_embedding_for_text()**
- **Purpose**: Generate embeddings for chunks using the Legal-BERT model.
- **Model Used**: `Legal-BERT`
- **Why Legal-BERT?**  
    - Trained on legal corpora, providing better contextual embeddings for legal texts.
    - **Alternatives**: `Sentence-BERT` or `OpenAI models` could offer alternatives but may not capture domain-specific nuances as well.

---

### 6. **prepare_faiss_and_clusters()**
- **Purpose**: Generate embeddings for chunks, create a FAISS index, and cluster chunks using K-Means.
- **Models Used**: 
    - **FAISS**: For fast similarity-based searches.
    - **K-Means**: For clustering chunks into meaningful groups.

- **Why FAISS and K-Means?**  
    - FAISS allows efficient similarity search, and K-Means ensures chunks are grouped logically.
    - **Alternatives**: HDBSCAN for clustering (non-parametric) and ElasticSearch for indexing.

**Improvement Suggestion**: Use dynamic clustering algorithms like HDBSCAN for more adaptable chunk grouping.

---

### 7. **dynamic_summary_mode()**
- **Purpose**: Handle query-based or full-document summarization.
- **Models Used**: K-Means for query-cluster prediction and FAISS for chunk retrieval.

- **Why this approach?**  
    - Separating query-based and full summarization ensures flexibility in responding to user needs.
    - **Alternatives**: Hybrid retrieval models that combine embeddings with traditional search techniques could be explored.

---

### 8. **generate_llama_summary_with_prompt()**
- **Purpose**: Generate an abstractive summary using the LLaMA model.
- **Why LLaMA?**  
    - LLaMA is a state-of-the-art LLM capable of abstractive summarization, providing more coherent outputs.
    - **Alternatives**: `GPT-4` could offer better performance but at a higher computational cost.

---

## Execution Flow

1. **Run `process_files()`**: Loads and preprocesses all PDFs in the specified directory.
2. **Prepare FAISS Index and K-Means Clusters**: Embeds chunks, creates the FAISS index, and clusters using K-Means.
3. **Handle Query or Full Summary Requests**:
   - If **query-based**, find the best matching cluster and retrieve relevant chunks.
   - If **full-document**, generate summaries for all chunks in each document.
4. **Generate and Return Summary**: Use LLaMA to generate an abstractive summary based on the query or document chunks.

---

## Future Improvements

1. **Asynchronous Processing**: Replace ThreadPoolExecutor with `asyncio` for better scalability.
2. **Domain-Specific Models**: Explore fine-tuning Legal-BERT on more specialized legal datasets.
3. **Advanced Chunking Strategies**: Implement adaptive or sliding window chunking to capture better context.
4. **Improved Clustering**: Use HDBSCAN for more dynamic clustering without needing to pre-define the number of clusters.

---

## Conclusion

This system efficiently processes and summarizes legal documents using state-of-the-art NLP and clustering techniques. With FAISS indexing, Legal-BERT embeddings, and LLaMA summarization, it provides both query-based and full-document summarization, ensuring relevant information retrieval for legal professionals.

---

## How to Run

1. Place the PDFs in the specified directory:  
   `"D:\AI_ML - PG\Capstone Project - Automated Legal document Segmentation\files_export\data\Documents\IP"`
2. Run the script to process files and generate summaries:

```bash
python your_script_name.py
```

3. Query Example:

```python
query = "Can a party terminate the contract without cause?"
dynamic_summary_mode(docs, faiss_index, clusters, kmeans_model, all_chunks, chunk_document_mapping, mode="query", query=query, top_k=2)
```
