Creating an HR Knowledge Base and Query System Using AWS Bedrock and LangChain
Overview
This project demonstrates how to create a knowledge base from a PDF document using AWS Bedrock embeddings and the LangChain library. The steps include loading and splitting a PDF, generating embeddings, creating a vector store, indexing the documents, and querying the indexed data using a language model.

# https://www.cesvi.eu/wp-content/uploads/2019/08/Human-Resources-Policy.pdf

# pip install flask-sqlalchemy

# pip install pypdf

# pip install faiss-cpu



Components and Flow
Import Libraries: Import the necessary libraries and modules.

Initialize Bedrock Client: Set up the AWS Bedrock client for creating embeddings.

Load PDF Data: Load and split the PDF document.

Text Splitting: Split the document into smaller chunks for better processing.

Create Vector Store and Index: Store embeddings in a vector database and create an index for efficient searching.

Connect to Bedrock LLM: Initialize the AWS Bedrock language model.

Query Function: Query the knowledge base using the language model.

Complete Pipeline Function: Execute the entire pipeline from loading the PDF to querying the knowledge base.

Usage Example: Demonstrate the usage of the complete pipeline with example queries.

# Code Explanation
python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings  # Ensure AWS credentials are configured
from langchain_community.vectorstores import FAISS  # Corrected import for FAISS
from langchain.vectorstores import VectorStore  # Importing VectorStore directly
from langchain.llms import Bedrock  # For Bedrock LLM interaction
from langchain.chains import RetrievalQA
import boto3

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your region
)

# 1. Configure Bedrock embeddings
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"
)
Imports: Import necessary modules.

Bedrock Client: Initialize the AWS Bedrock client.

python
# 2. Load and split the PDF
def load_pdf_data(pdf_url):
    loader = PyPDFLoader(pdf_url)
    documents = loader.load_and_split()
    print(f"Number of documents loaded: {len(documents)}")
    return documents
Load PDF Data: Function to load and split a PDF document using PyPDFLoader.

python
# 3. Text splitting configuration
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Number of split documents: {len(split_docs)}")
    return split_docs
Text Splitting: Function to split the text of the documents into smaller chunks for processing.

python
# 4 & 5. Create vector store and index
def create_vector_index(split_docs):
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    # No need for VectorStoreIndexCreator, directly use vectorstore
    return vectorstore
Create Vector Store and Index: Function to create a vector store from the split documents and generate embeddings.

python
# 6a. Connect to Bedrock LLM
def initialize_bedrock_llm(model_id="amazon.titan-text-express-v1"):
    llm = Bedrock(
        client=bedrock_client,
        model_id=model_id
    )
    return llm
Connect to Bedrock LLM: Function to initialize the Bedrock language model.

python
# 6b. Query function
def query_document(query, vectorstore, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
    )
    return qa_chain.invoke(query)  # Use the invoke method instead of run
Query Function: Function to query the vector store and use the Bedrock language model to get the answer.

python
# Complete pipeline function
def create_hr_knowledge_base(pdf_url):
    # Load and split documents
    raw_documents = load_pdf_data(pdf_url)
    
    # Split text
    split_documents = split_text(raw_documents)
    
    # Create vector store
    vectorstore = create_vector_index(split_documents)
    
    # Initialize LLM
    llm = initialize_bedrock_llm()
    
    return vectorstore, llm
Complete Pipeline: Function to create the knowledge base by loading, splitting, embedding, and indexing the PDF document.

python
# Usage example
def main():
    pdf_url = 'https://www.cesvi.eu/wp-content/uploads/2019/08/Human-Resources-Policy.pdf'
    
    # Create knowledge base
    vectorstore, llm = create_hr_knowledge_base(pdf_url)
    
    # Example queries
    queries = [
        "What is the policy regarding sick leave?",
        "How many casual leaves are allowed?"
    ]
    
    # Run queries
    for query in queries:
        print(f"\nQuestion: {query}")
        answer = query_document(query, vectorstore, llm)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
Usage Example: Main function to create the knowledge base from a specified PDF URL and run example queries.

Usage Instructions
Ensure AWS Credentials: Make sure your AWS credentials are configured and have access to AWS Bedrock.

Install Dependencies: Install the necessary Python packages:

bash
pip install boto3 langchain langchain_community
Run the Script: Execute the script to create the knowledge base and run queries:

bash
