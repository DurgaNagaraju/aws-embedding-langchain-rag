import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings  # Ensure AWS credentials are configured
from langchain_community.vectorstores import FAISS  # Corrected import for FAISS
from langchain.indexes import VectorStoreIndexCreator  # Corrected import for VectorStoreIndexCreator
from langchain.llms import Bedrock  # For Bedrock LLM interaction
from langchain.chains import RetrievalQA

# Define the data source and load data
data_load = PyPDFLoader('https://www.cesvi.eu/wp-content/uploads/2019/08/Human-Resources-Policy.pdf')
documents = data_load.load_and_split()
print(f"Number of documents loaded: {len(documents)}")
print(documents[0])

# Split the text based on characters, tokens, etc.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust as needed
    chunk_overlap=200,  # Adjust as needed
    separators=["\n\n", "\n", " ", ""]
)
split_documents = text_splitter.split_documents(documents)
print(f"Number of documents after splitting: {len(split_documents)}")

# Create embeddings with AWS Bedrock
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")  # Ensure AWS credentials are configured

# Create Vector DB and store embeddings
db = FAISS.from_documents(split_documents, embeddings)  # Using FAISS for vector store

# Create index for search
index = VectorStoreIndexCreator().from_vectorstore(db)

# Wrap index creation in a function
def create_hr_index(pdf_url):
    loader = PyPDFLoader(pdf_url)
    docs = loader.load_and_split()
    split_docs = text_splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embeddings)  # Using FAISS for vector store
    index = VectorStoreIndexCreator().from_vectorstore(db)
    return index

# Function to connect to Bedrock Foundation Model
def connect_to_bedrock(model_id):  # Example: "titan-text-g1-express"
    llm = Bedrock(model_id=model_id)  # Add region_name if needed
    return llm

# Function to query the HR document and interact with Bedrock LLM
def query_hr_document(query, index, llm):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever())  # Using "stuff" chain type
    result = qa_chain.run(query)
    return result

# Example Usage
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
