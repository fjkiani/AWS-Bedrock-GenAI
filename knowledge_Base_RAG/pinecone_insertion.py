import os
import boto3
import PyPDF2
import cohere
from pinecone import Pinecone, ServerlessSpec
import json
import uuid

# Function to get Pinecone API key from AWS Secrets Manager
def get_secret(secret_name, region_name="us-west-2"):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        raise e
    
    secret = get_secret_value_response["SecretString"]
    secret_dict = json.loads(secret)
    return secret_dict["pinecone_api_key"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to generate embeddings using Cohere
def generate_embeddings(text):
    cohere_client = cohere.Client('')
    response = cohere_client.embed(
        texts=[text],
        model='large',  # Use the appropriate model name
    )
    embeddings = response.embeddings[0]
    return embeddings

# Extract text from the PDF
pdf_path = '/Users/fahadkiani/Desktop/development/Bedrock/knowledge_Base_RAG/files/Description+of+the+various+ErrorCode+in+New+Account+Opening.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Generate embeddings from the extracted text
pdf_embedding = generate_embeddings(pdf_text)

# Retrieve Pinecone API key from AWS Secrets Manager
secret_name = "arn:aws:secretsmanager:us-west-2:312708185285:secret:pinecone/credentials/1989-ehUyZ1"
pinecone_api_key = get_secret(secret_name)

# Setup Pinecone connection
pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to the Pinecone index
index_name = 'testing-index'  # Use a valid index name
embedding_dimension = 4096  # Set the dimension to 4096

# Check if the index exists and delete it if necessary
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=embedding_dimension,  # Ensure the dimension matches your embedding size
    metric='euclidean',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'  # Change to a supported region
    )
)

index = pc.Index(index_name)

# Define metadata fields
text_field_name = "text_field"
metadata_field_name_source = "source"
metadata_field_name_document_id = "document_id"

# Insert the embedding into Pinecone with metadata
record_id = str(uuid.uuid4())
metadata = {
    metadata_field_name_source: "example_source",
    metadata_field_name_document_id: record_id,
    text_field_name: pdf_text
}
index.upsert([(record_id, pdf_embedding, metadata)])

print("PDF text and embedding inserted into Pinecone successfully.")
