
# Automated Legal Document Segmentation and Summarization

## Overview

This project performs **automated legal document segmentation, query-based retrieval, and summarization**. The system uses **Legal-BERT** embeddings for semantic understanding, **Named Entity Recognition (NER)** to extract important entities, **FAISS** for efficient similarity search, **K-Means clustering** for grouping, and **LLaMA** for abstractive summarization.

---

## **1. Process Flow of the Code**

### **Extracting and Preprocessing PDF Files**
- Extracts text from PDFs, normalizes it, and extracts named entities using **Spacy’s NER model**.
- **Legal-BERT embeddings** generate vector representations of chunks for similarity search and categorization.

```python
def process_file(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "
".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        normalized_text = text_normalization(full_text)
        named_entities = named_entity_recognition(normalized_text)
        chunks = nlp_based_chunking(normalized_text)
        return {"file_name": os.path.basename(pdf_path), "normalized_text": normalized_text, "named_entities": named_entities, "chunks": chunks}
```

---

## **2. Text Normalization and Named Entity Recognition**
- **Text normalization:** Converts text to lowercase, removes punctuation, and trims extra spaces.
- **NER:** Extracts structured information like dates, names, and locations using Spacy.

```python
def text_normalization(text):
    return re.sub(r'\s+', ' ', text.lower().translate(str.maketrans("", "", string.punctuation))).strip()

def named_entity_recognition(text):
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]
```

---

## **3. NLP-Based Chunking and Categorization**
- Chunks text into sentences and categorizes them using **Legal-BERT embeddings**.
- **Cosine similarity** identifies the closest legal category for each sentence.

```python
def nlp_based_chunking(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [{"category": categorize_sentence(sentence), "text": sentence.strip()} for sentence in sentences]
```

---

## **4. Embedding Storage and Clustering Using FAISS and K-Means**
- **FAISS:** Stores document embeddings for similarity search.
- **K-Means:** Groups chunks into clusters to improve query-based retrieval.

```python
def prepare_faiss_and_clusters(docs, num_clusters):
    all_chunks = [chunk for doc in docs for chunk in doc['chunks']]
    embeddings = get_document_embeddings_batch(all_chunks)
    faiss_index = initialize_faiss_index(embeddings.shape[1])
    add_embeddings_to_faiss(faiss_index, embeddings)
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans_model.fit_predict(embeddings)
    return faiss_index, embeddings, clusters, kmeans_model, all_chunks
```

---

## **5. Query Handling and Retrieval with FAISS**
- **K-Means identifies relevant clusters** for a query.
- **FAISS performs similarity searches** within the identified cluster.

```python
def dynamic_summary_mode(docs, faiss_index, clusters, kmeans_model, all_chunks, query=None):
    query_embedding = preprocess_query(query)
    query_cluster = kmeans_model.predict(query_embedding)[0]
    relevant_indices = [i for i, cluster in enumerate(clusters) if cluster == query_cluster]
    distances, indices = search_faiss(faiss_index, query_embedding)
```

---

## **6. Abstractive Summarization Using LLaMA**
- Generates **abstractive summaries** focused on query-specific information using **prompt engineering**.

```python
def generate_llama_summary_with_prompt(chunk_text, named_entities, matched_categories, query=None):
    data = {"model": "llama3.1", "prompt": f"Extract information related to the query '{query}'.

{chunk_text}"}
    response = requests.post("http://127.0.0.1:11434/api/generate", json=data)
    return "".join(line.decode('utf-8') for line in response.iter_lines())
```

---

## **7. Complete Process Flow Summary**
1. **Extract PDF Text:** Extracts and normalizes text from PDFs.
2. **NER:** Identifies structured entities like dates and organizations.
3. **Chunking and Categorization:** Breaks documents into chunks categorized by legal topics.
4. **FAISS and K-Means:** Stores embeddings and groups them into clusters.
5. **Query Handling:** Retrieves relevant chunks using similarity search.
6. **Summarization:** LLaMA generates concise query-based summaries.

---

## **Usage**
1. **Install Dependencies:**
   ```
   pip install pdfplumber spacy numpy faiss-cpu transformers scikit-learn tensorflow requests
   python -m spacy download en_core_web_sm
   ```

2. **Prepare PDF Files:**
   - Place the PDF files in the specified directory.

3. **Run the Code:**
   Update the directory path in the script:
   ```python
   pdf_directory = "path_to_your_pdf_files"
   docs = process_files(pdf_directory)
   ```

4. **Execute Query:**
   Use `dynamic_summary_mode` to retrieve relevant summaries:
   ```python
   query = "Can a party terminate the contract without cause?"
   dynamic_summary_mode(docs, faiss_index, clusters, kmeans_model, all_chunks, query=query)
   ```

---

## **Conclusion**
This system ensures **efficient legal document retrieval and summarization** using **state-of-the-art NLP models**, **FAISS-based search**, and **K-Means clustering**. The integration of **LLaMA** for abstractive summarization further enhances the system's ability to provide concise and relevant summaries tailored to user queries.
