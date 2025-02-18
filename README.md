# aws-embedding-langchain-rag
https://www.cesvi.eu/wp-content/uploads/2019/08/Human-Resources-Policy.pdf

pip install flask-sqlalchemy

pip install pypdf

pip install faiss-cpu

Step-by-Step Pipeline for AWS Bedrock Embedding Model:
1. Import Modules
python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings  # Ensure AWS credentials are configured
from langchain.vectorstores import FAISS  # Using FAISS for vector store
from langchain.indexes.vectorstore import VectorStoreIndexCreator
from langchain.llms import Bedrock  # For Bedrock LLM interaction
from langchain.chains import RetrievalQA
2. Define Data Source and Load Data
python
# Define the data source
data_load = PyPDFLoader('https://www.cesvi.eu/wp-content/uploads/2019/08/Human-Resources-Policy.pdf')
# Load and split the document
documents = data_load.load_and_split()
print(f"Number of documents loaded: {len(documents)}")
print(documents[0])
3. Split the Text
python
# Split the text based on characters, tokens, etc.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust as needed
    chunk_overlap=200,  # Adjust as needed
    separators=["\n\n", "\n", " ", ""]
)
split_documents = text_splitter.split_documents(documents)
print(f"Number of documents after splitting: {len(split_documents)}")
4. Create Embeddings
python
# Create embeddings with AWS Bedrock
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")  # Ensure AWS credentials are configured
5a. Create Vector DB, Store Embeddings, and Index
python
# Create Vector DB and store embeddings
db = FAISS.from_documents(split_documents, embeddings)  # Using FAISS for vector store

# Create index for search
index = VectorStoreIndexCreator().from_vectorstore(db)
5b. Create Index for HR Report
This step is part of the previous code block, where we already created the index for search.

5c. Wrap within a Function
python
# Wrap index creation in a function
def create_hr_index(pdf_url):
    loader = PyPDFLoader(pdf_url)
    docs = loader.load_and_split()
    split_docs = text_splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embeddings)  # Using FAISS for vector store
    index = VectorStoreIndexCreator().from_vectorstore(db)
    return index
6a. Write a Function to Connect to Bedrock Foundation Model
python
# Function to connect to Bedrock Foundation Model
def connect_to_bedrock(model_id):  # Example: "titan-text-g1-express"
    llm = Bedrock(model_id=model_id)  # Add region_name if needed
    return llm
6b. Write a Function for Search and LLM Interaction
python
# Function to query the HR document and interact with Bedrock LLM
def query_hr_document(query, index, llm):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever())  # Using "stuff" chain type
    result = qa_chain.run(query)
    return result
Example Usage
python
# Create index for the HR document
hr_index = create_hr_index('https://www.cesvi.eu/wp-content/uploads/2019/08/Human-Resources-Policy.pdf')

# Connect to the Bedrock model
bedrock_llm = connect_to_bedrock("titan-text-g1-express")  # Replace with your desired Bedrock model ID

# User query example 1
user_query = "What is the policy regarding sick leave?"
answer = query_hr_document(user_query, hr_index, bedrock_llm)
print(answer)

# User query example 2
user_query = "How many casual leaves are allowed?"
answer = query_hr_document(user_query, hr_index, bedrock_llm)
print(answer)

# Example of using a different PDF and re-indexing
# new_hr_index = create_hr_index("another_pdf_url.pdf")
# new_answer = query_hr_document("What about maternity leave?", new_hr_index, bedrock_llm)
# print(new_answer)
Summary
Import Modules: Import necessary libraries and modules.

Load Data: Load the PDF document.

Text Splitting: Split the document into smaller chunks.

Create Embeddings: Generate embeddings using AWS Bedrock.

Create Vector DB and Index: Store embeddings in a vector DB and create an index.

Connect to Bedrock LLM: Function to connect to the AWS Bedrock model.

Query and Interaction: Function to query the indexed document and interact with the LLM.

This pipeline sets up the entire process from loading and processing the document to querying it and interacting with the language model. Make sure your AWS credentials are configured properly and you have access to the Bedrock models. Let me know if you have any further questions or need additional assistance!
