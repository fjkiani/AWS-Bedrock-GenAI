import cohere

def generate_embeddings(text):
    cohere_client = cohere.Client('##')
    response = cohere_client.embed(
        texts=[text],
        model='large',  # Example model, choose as per your need
    )
    embeddings = response.embeddings[0]
    print(f"Embedding dimension: {len(embeddings)}")  # Print embedding dimension
    return embeddings

# Example text
text = "This is a test text."
embedding = generate_embeddings(text)
