
# Automated Legal Document Summarization


The goal of this code is to **automate the extraction, categorization, and summarization of legal documents** using **machine learning models (Legal-BERT, K-Means)** and **efficient search engines (FAISS)**. The code allows you to:
1. **Process legal documents stored as PDFs**.
2. **Identify and categorize legal clauses** using **NLP techniques**.
3. **Generate embeddings** for text chunks to store them in a **FAISS index**.
4. **Cluster similar chunks using K-Means** to make **query-based retrieval faster and more accurate**.
5. Use **dynamic query summarization** (with **LLaMA**) to provide **abstractive answers**.
6. **Improve the user experience** by allowing questions about specific legal clauses and retrieving **only relevant sections**.

This solution ensures **faster document comprehension** and **precise answers** to user queries, reducing the effort needed to manually scan through complex legal texts.

---

## Detailed Execution Flow:

### Step 1: Process Legal Documents from PDFs
**What Happens?**
1. **Function: `process_files()`**
   - This function takes a **folder path containing PDF files**.
   - It **loads and processes each PDF in parallel** using a **ThreadPoolExecutor**, ensuring **faster execution**.
   
**Each PDF Goes Through:**
   - **`process_file()`** extracts the **full text** from the PDF using **pdfplumber**.
   - The text is **normalized** to remove punctuation and unnecessary spaces.
   - **Named Entity Recognition (NER)** is applied to **extract key entities** like dates and names.
   - **NLP-based chunking** is performed to split the text into smaller chunks aligned with **legal categories**.

**Impact:**  
- Efficient handling of multiple PDFs in parallel.  
- Text normalization and chunking allow for **better processing and categorization** of legal clauses.

---

### Step 2: Chunking and Categorization of Legal Text
**What Happens?**
1. **Function: `nlp_based_chunking()`**
   - Splits the text into sentences.
   - **Categorizes each sentence** by finding the **closest matching category** from the **41 legal categories** (like **Termination Clause, Warranty, etc.**).
   - If no category matches, the sentence is labeled as **"General."**

2. **How Categories Are Assigned:**
   - Each sentence is **embedded using Legal-BERT**.
   - The embedding is **compared with pre-stored category embeddings** using **cosine similarity**.
   - If the similarity exceeds **0.7**, the sentence is assigned to that category.

**Impact:**  
- Accurate categorization helps in **efficient retrieval** of specific legal clauses during query processing.  
- Using **Legal-BERT embeddings** ensures that the system understands **legal-specific vocabulary**.

---

### Step 3: Embedding Generation and FAISS Index Creation
**What Happens?**
1. **Function: `get_document_embeddings_batch()`**
   - Processes chunks in **batches** to avoid memory overload.
   - Each chunk is converted into **numerical embeddings** using **Legal-BERT**.
   
2. **Function: `initialize_faiss_index()` and `add_embeddings_to_faiss()`**
   - Initializes a **FAISS index** to store embeddings.
   - **FAISS** enables **fast similarity searches** within the embeddings.

**Impact:**  
- Embeddings make it possible to perform **semantic search** across chunks, going beyond keyword matching.  
- **FAISS indexing** ensures **fast retrieval**, even with large collections of documents.

---

### Step 4: K-Means Clustering of Document Chunks
**What Happens?**
1. **Function: `prepare_faiss_and_clusters()`**
   - Flattens all chunks into a single list.
   - **Generates embeddings** for all chunks and stores them in **FAISS**.
   - **K-Means clustering** is applied to group **similar chunks** into clusters.

2. **Why Clustering Matters:**  
   - During **query processing**, the system predicts **which cluster** the query belongs to.
   - Only chunks from the predicted cluster are searched, reducing the **search space** and **improving speed**.

**Impact:**  
- Clustering helps **narrow down the search space**, leading to **faster query responses**.  
- **K-Means** ensures that **semantically similar clauses** are grouped together.

---

### Step 5: Query Processing and Dynamic Summarization
**What Happens?**
1. **Function: `dynamic_summary_mode()`**
   - Accepts a **query** and preprocesses it to generate an **embedding** using **Legal-BERT**.
   - The query’s embedding is used to **predict the most relevant cluster** using the **K-Means model**.
   - The relevant chunks are **searched within the FAISS index** for further matching.

2. **If a Query Matches Multiple Categories:**
   - The system matches the query to **multiple legal categories** (if applicable).
   - Chunks related to those categories are prioritized in the **retrieval process**.

3. **Abstractive Summarization:**
   - If the query requires **detailed answers**, the **LLaMA model** generates an **abstractive summary**.
   - A **specific prompt** is created to focus on relevant legal clauses.

**Impact:**  
- Query-based retrieval ensures **only relevant information** is presented, saving users from scanning entire documents.  
- **LLaMA’s abstractive summarization** provides **concise summaries** tailored to the query’s intent.

---

### Step 6: Running the System and Demonstrating the Flow
1. **PDFs are processed** using `process_files()`.
2. **Chunks and embeddings are generated** and stored in **FAISS**.
3. **K-Means clustering groups chunks** for efficient search.
4. A **query is processed** through `dynamic_summary_mode()`:
   - **Predicts the relevant cluster**.
   - **Retrieves relevant chunks** from the FAISS index.
   - **Generates a summary or answers the query** using LLaMA.

---

## How Effective Is This Approach?

1. **Scalable and Efficient Search:**  
   - **FAISS** ensures fast **similarity searches**, even with large document sets.
   - **K-Means clustering** reduces the **search space**, improving **query response times**.

2. **Accurate Categorization:**  
   - Using **Legal-BERT embeddings** and **41 legal categories** ensures that the **retrieved information aligns with legal-specific queries**.

3. **Parallel Processing:**  
   - **ThreadPoolExecutor** speeds up the **PDF loading and processing** stages.

4. **Dynamic Query Handling:**  
   - The system can **handle complex legal queries** and **provide both extractive and abstractive summaries**.

5. **Customizable and Extendable:**  
   - The system can easily be **extended** by adding more **legal categories** or **fine-tuning models** for specific use cases.

---

## Limitations and Future Improvements
1. **Memory Usage:**  
   - Embedding large documents may require **more memory**.

2. **LLaMA Inference Speed:**  
   - **Abstractive summarization** using LLaMA might be **slow**. Consider **GPU-based inference**.

3. **Improved Clustering:**  
   - **HDBSCAN** could be explored for **non-uniform clusters**.

---

## Summary of What You Achieve
- **Automated PDF processing** with legal clause extraction and categorization.
- **Efficient query-based retrieval** using FAISS and K-Means clustering.
- **Dynamic summarization** through **LLaMA** tailored to specific queries.
- **Scalable solution** that accelerates legal document review.
