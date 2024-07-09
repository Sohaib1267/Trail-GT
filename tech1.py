import numpy as np
import faiss

documents = [
    "Document 1 content here",
    "Content of the second document",
    "The third one has different content",
]
metadata = [
    {"date": "20230101", "tag": "news"},
    {"date": "20230102", "tag": "update"},
    {"date": "20230103", "tag": "report"},
]

# Dummy function to generate embeddings
def generate_embeddings(texts):
    """Generate dummy embeddings for the sake of example."""
    return np.random.rand(len(texts), 128).astype('float32')  # 128-dimensional embeddings

# Generate embeddings for documents
doc_embeddings = generate_embeddings(documents)

# Create a FAISS index for the embeddings (using FlatL2 for simplicity)
index = faiss.IndexFlatL2(128)  # 128 is the dimensionality of the vectors
index.add(doc_embeddings)  # Add embeddings to the index

# Example search function that uses metadata
def search(query_embedding, metadata_key, metadata_value):
    """Search the index for documents that match metadata criteria."""
    k = 2  # Number of nearest neighbors to find
    distances, indices = index.search(np.array([query_embedding]), k)  # Perform the search
    results = []
    for idx in indices[0]:
        if metadata[idx][metadata_key] == metadata_value:
            results.append((documents[idx], metadata[idx]))
    return results

# Generate a query embedding (in a real scenario, this would come from a similar process)
query_embedding = generate_embeddings(["data science"])[0]

# Search for documents tagged with 'update'
matching_documents = search(query_embedding, 'tag', 'update')
print(matching_documents)