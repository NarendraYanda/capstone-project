import os
import pdfplumber
import logging
import asyncio
import concurrent.futures
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from sklearn.metrics import precision_recall_fscore_support
import chainlit as cl
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Rouge for summarization evaluation
rouge = Rouge()

# Load Sentence Transformer model for BERT-based embeddings
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Global variable to store original docs
original_docs = []

# Parallel loading and splitting of PDFs
def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    try:
        logging.info(f"Loading and splitting PDF: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            doc = Document(page_content=text, metadata={"file_name": os.path.basename(file_path)})
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_text(doc.page_content)
            logging.info(f"File {file_path} split into {len(chunks)} chunks.")
            return [Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks]
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []

# Load PDFs from folder in parallel with extra logging
def load_pdfs_from_folder_parallel(folder_path, chunk_size=1000, chunk_overlap=200):
    documents = []
    global original_docs  # Keep track of the original documents
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        logging.info(f"Looking for PDF files in: {folder_path}")
        for file_name in os.listdir(folder_path):
            logging.info(f"Found file: {file_name}")
            if file_name.lower().endswith(".pdf"):  # Convert to lowercase for case-insensitive check
                file_path = os.path.join(folder_path, file_name)
                logging.info(f"Processing file: {file_path}")
                futures.append(executor.submit(load_and_split_pdf, file_path, chunk_size, chunk_overlap))
            else:
                logging.warning(f"File {file_name} is not a PDF and will be skipped.")

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                documents.extend(result)
                original_docs.extend(result)  # Store original docs for metadata access
    
    logging.info(f"Total documents (chunks) loaded: {len(documents)}")
    return documents

# Function to format documents into string format for the model
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# BERT-based extractive summarization
def bert_extractive_summary(normalized_text, max_sentences=3):
    logging.info(f"Performing BERT extractive summarization.")
    sentences = normalized_text.split(". ")
    sentence_embeddings = bert_model.encode(sentences)
    document_embedding = bert_model.encode([normalized_text])[0]
    similarity_scores = cosine_similarity(sentence_embeddings, document_embedding.reshape(1, -1)).flatten()
    ranked_sentences = [sentences[i] for i in similarity_scores.argsort()[::-1]]
    logging.info(f"Extractive summarization completed.")
    return " ".join(ranked_sentences[:max_sentences])

# Normalize the document names (remove special characters and normalize spaces)
def normalize_document_name(doc_name):
    return re.sub(r'[^a-zA-Z0-9 ]', '', doc_name).replace('_', ' ').strip().lower()

# Initialize embeddings, vectorstore, and the LLM model with logging
async def initialize_vectorstore_and_model(all_splits):
    try:
        logging.info("Initializing embeddings and vectorstore...")
        
        # Initialize local embeddings (smaller embedding model for faster processing)
        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Create vectorstore using FAISS
        vectorstore = await cl.make_async(FAISS.from_documents)(all_splits, embedding=local_embeddings)

        # Define LLM model (switch to a smaller, faster model if possible)
        model = ChatOllama(model="llama2:7b")

        logging.info("Vectorstore and model initialized successfully.")
        
        # Print unique document names stored in vectorstore
        unique_document_names = set([doc.metadata['file_name'] for doc in original_docs])
        logging.info("Printing unique document names stored in vectorstore:")
        for file_name in unique_document_names:
            logging.info(f"Stored document name: {file_name}")
        
        return vectorstore, model
    except Exception as e:
        logging.error(f"Error initializing vectorstore and model: {e}")
        return None, None

# Function for RAG Chain-based summarization with logging and error handling
async def summarize_docs_with_rag_chain(docs, model):
    try:
        logging.info("Running summarization...")

        prompt = ChatPromptTemplate.from_template("Summarize the main themes in these retrieved docs: {docs}")
        chain = {"docs": format_docs} | prompt | model | StrOutputParser()

        logging.info(f"Docs passed for summarization: {docs}")
        
        # Make async summarization call using cl.make_async to avoid blocking
        summary = await cl.make_async(chain.invoke)(docs)

        if summary and len(summary) > 0:
            logging.info(f"Summary generated: {summary[:100]}...")
            return summary
        else:
            logging.warning("No valid summary generated.")
            return "No valid summary generated."
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return "Error during summarization."

# Function for RAG Chain-based QA with timeout and logging
async def qa_with_rag_chain(docs, question, model):
    try:
        logging.info("Running QA...")

        RAG_TEMPLATE = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        <context>
        {context}
        </context>

        Answer the following question:

        {question}"""

        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | model
            | StrOutputParser()
        )

        # Log context size and handle QA with a 30s timeout
        logging.info(f"Context size for QA: {len(format_docs(docs))} characters")
        answer = await cl.make_async(chain.invoke)({"context": docs, "question": question})
        return answer
    except asyncio.TimeoutError:
        logging.warning("QA task timed out.")
        return "QA task timed out."
    except Exception as e:
        logging.error(f"Error during QA: {e}")
        return "Error during QA."

# Ask user for PDF directory and interact via Chainlit
@cl.on_chat_start
async def chat_start():
    welcome_message = """
    Hello! Welcome to the PDF Summarization and QA assistant.
    
    You can perform the following actions:
    1. **Summarize all documents**: Simply ask to summarize all loaded documents.
    2. **Summarize a specific document**: For example, type `summarize ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT`.
    3. **BERT Extractive summarization**: Use BERT to extract key sentences. For example, `extractive summarize ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT`.
    4. **QA for a specific document**: Ask a question like, `What are the terms of termination in ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT?`.
    
    Please upload the documents by specifying the folder path as `pdfdir: your_folder_path`.
    """
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def handle_message(message):
    global vectorstore, model

    if message.content.startswith("pdfdir:"):
        folder_path = message.content.split("pdfdir:", 1)[1].strip()

        if os.path.isdir(folder_path):
            logging.info(f"Loading PDFs from directory: {folder_path}")
            documents = load_pdfs_from_folder_parallel(folder_path)

            if documents:
                vectorstore, model = await initialize_vectorstore_and_model(documents)
                await cl.Message(content="PDFs have been loaded and processed. Please enter a query for summarization or QA.").send()
            else:
                await cl.Message(content="No documents found in the directory or processing error occurred.").send()
        else:
            await cl.Message(content="Invalid directory. Please provide a valid folder path.").send()

    elif message.content:
        # Check if vectorstore and model are initialized
        if vectorstore is None or model is None:
            await cl.Message(content="Vectorstore or model not initialized. Please load PDFs first.").send()
            return

        # Handle specific document summarization
        if message.content.lower().startswith("summarize"):
            logging.info(f"Running summarization for query: {message.content}")
            # Check if it's a specific document summarization
            doc_name = message.content.lower().split("summarize ", 1)[1].strip()
            normalized_doc_name = normalize_document_name(doc_name)
            matching_docs = [doc for doc in original_docs if normalize_document_name(doc.metadata['file_name']) == normalized_doc_name]

            if matching_docs:
                summary = await summarize_docs_with_rag_chain(matching_docs, model)

                # Check if summary is valid before sending
                if summary and len(summary) > 0:
                    logging.info(f"Sending summary back to Chainlit: {summary[:100]}...")
                    await cl.Message(content=summary).send()
                    logging.info("Summary sent to Chainlit successfully.")
                else:
                    logging.warning("Summary not generated, sending fallback message.")
                    await cl.Message(content="No summary was generated.").send()
            else:
                logging.warning(f"No relevant documents found with the name '{normalized_doc_name}' for summarization.")
                await cl.Message(content=f"No relevant documents found with the name '{normalized_doc_name}' for summarization.").send()

        # Handle Extractive Summarization using BERT
        elif message.content.lower().startswith("extractive summarize"):
            logging.info(f"Running BERT extractive summarization for query: {message.content}")
            doc_name = message.content.lower().split("extractive summarize", 1)[1].strip()
            normalized_doc_name = normalize_document_name(doc_name)
            matching_docs = [doc for doc in original_docs if normalize_document_name(doc.metadata['file_name']) == normalized_doc_name]

            if matching_docs:
                extractive_summary = bert_extractive_summary(format_docs(matching_docs))
                if extractive_summary and len(extractive_summary) > 0:
                    await cl.Message(content=extractive_summary).send()
                else:
                    logging.warning("Extractive summary not generated.")
                    await cl.Message(content="No extractive summary generated.").send()
            else:
                await cl.Message(content=f"No relevant documents found with the name '{normalized_doc_name}' for extractive summarization.").send()

        # Handle QA
        elif message.content.lower().startswith("qa") or "?" in message.content:
            logging.info(f"Running QA for query: {message.content}")

            # Check if it's for a specific document
            if " in " in message.content.lower():
                # Extract document name from the query
                doc_name = message.content.lower().split(" in ", 1)[1].strip().rstrip("?")
                normalized_doc_name = normalize_document_name(doc_name)
                matching_docs = [doc for doc in original_docs if normalize_document_name(doc.metadata['file_name']) == normalized_doc_name]

                if matching_docs:
                    answer = await qa_with_rag_chain(matching_docs, message.content, model)

                    # Send QA result back to Chainlit
                    if answer and len(answer) > 0:
                        logging.info(f"Sending QA result back to Chainlit: {answer[:100]}...")
                        await cl.Message(content=answer).send()
                    else:
                        logging.warning("No answer generated, sending fallback message.")
                        await cl.Message(content="No answer was generated.").send()
                else:
                    logging.warning(f"No relevant documents found with the name '{normalized_doc_name}' for QA.")
                    await cl.Message(content=f"No relevant documents found with the name '{normalized_doc_name}' for QA.").send()

            else:
                # Look through all documents if no specific document is mentioned
                docs = vectorstore.similarity_search(message.content)

                if docs:
                    answer = await qa_with_rag_chain(docs, message.content, model)

                    # Send QA result back to Chainlit
                    if answer and len(answer) > 0:
                        logging.info(f"Sending QA result back to Chainlit: {answer[:100]}...")
                        await cl.Message(content=answer).send()
                    else:
                        logging.warning("No answer generated, sending fallback message.")
                        await cl.Message(content="No answer was generated.").send()
                else:
                    await cl.Message(content="No relevant documents found for QA.").send()
