
## **Features**
- **Parallel PDF Processing**: Uses `pdfplumber` to extract text from PDFs and process them concurrently.
- **Embedding Storage with FAISS**: Enables fast similarity search across large document collections.
- **BERT-based Extractive Summarization**: Generates concise summaries based on sentence-level importance.
- **RAG Chain-based Query Answering**: Combines retrieval with generation using large language models (LLMs).
- **Chainlit Interface**: Provides a user-friendly interface to interact with the summarization and query system.

## **Installation**
To run the code, ensure you have Python 3.8+ installed and set up the following dependencies:

```bash
pip install pdfplumber spacy faiss-cpu requests sentence-transformers langchain rouge matplotlib seaborn chainlit
python -m spacy download en_core_web_sm
```

## **How It Works**

### 1. **PDF Loading and Chunking**
PDFs are loaded and split into chunks using `RecursiveCharacterTextSplitter`. Each chunk overlaps to ensure context retention across sections.

### 2. **FAISS Vectorstore Initialization**
The chunks are embedded using a BERT model, and the embeddings are stored in a FAISS vectorstore for fast similarity searches.

### 3. **RAG-based Summarization and Query Handling**
RAG chains combine retrieval (from FAISS) and generation (using LLMs) to provide accurate summaries and query answers.

### 4. **User Interaction with Chainlit**
Chainlit handles user interactions, allowing users to upload PDFs, request summaries, and ask queries.

## **Execution Flow**

1. **Loading PDFs in Parallel**:
   - `load_and_split_pdf()`: Extracts text from PDFs and splits them into overlapping chunks.
   - `load_pdfs_from_folder_parallel()`: Loads multiple PDFs concurrently.

2. **Embedding and Vectorstore Setup**:
   - `initialize_vectorstore_and_model()`: Initializes FAISS vectorstore with embeddings and loads the LLaMA model.

3. **Summarization and Query Handling**:
   - `summarize_docs_with_rag_chain()`: Summarizes documents using RAG chains.
   - `qa_with_rag_chain()`: Answers user queries with contextual information from PDFs.

4. **Chainlit Interaction**:
   - `@cl.on_chat_start`: Displays a welcome message and prompts for PDF upload.
   - `@cl.on_message`: Handles user inputs for summarization and queries.

## **Example Usage**
Place your PDFs in a folder and provide the path as input to the system:

In the Chainlit interface, you can:
- **Summarize a document**: Type `summarize <document_name>`
- **Ask a query**: Type your question directly.

## **Future Improvements**
- **Scalability**: Explore Pinecone or Weaviate for better scalability.
- **Multi-lingual Support**: Integrate language detection and multi-lingual models.
- **Active Learning**: Implement feedback loops to refine the system based on user inputs.

## **Acknowledgments**
- **LangChain** for RAG chains and prompt templates.
- **FAISS** for fast vector similarity search.
- **Sentence Transformers** for BERT-based embeddings.
- **Chainlit** for the user interface.
