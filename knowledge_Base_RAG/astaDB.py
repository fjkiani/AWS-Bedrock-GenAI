import PyPDF2
import cohere
from astrapy.client import DataAPIClient
import uuid

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
    cohere_client = cohere.Client('##')
    response = cohere_client.embed(
        texts=[text],
        model='large',  # Example model, choose as per your need
    )
    return response.embeddings[0][:1000]  # Reduce the size of the embedding to 1000 elements

# Extract text from the PDF
pdf_path = '/Users/fahadkiani/Desktop/development/Bedrock/knowledge_Base_RAG/files/Description+of+the+various+ErrorCode+in+New+Account+Opening.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Generate embeddings from the extracted text
pdf_embedding = generate_embeddings(pdf_text)

# Setup AstraDB connection using astrapy
client = DataAPIClient("AstraCS:h##")
db = client.get_database_by_api_endpoint(
    "https://94fbc102-f833-43e1-8a95-23fd0774fda1-us-east-2.apps.astra.datastax.com",
    namespace="testing"
)

# Specify the collection (table equivalent in AstraDB)
collection = db.get_collection("article_embeddings")

# Insert the embedding into AstraDB
record = {
    "id": str(uuid.uuid4()),
    "account_id": "unique-id-for-your-pdf",
    "embedding": pdf_embedding
}

collection.insert_one(record)

print("PDF text and embedding inserted into AstraDB successfully.")
