import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path, embedding_model_name):
        """
        Initialize the retriever with paths to the FAISS index and metadata.

        Args:
            index_path (str): Path to the FAISS index file.
            embedding_model_name (str): Hugging Face model name for embeddings.
        """
        # Load FAISS index
        
        self.index = faiss.read_index(index_path)

        # Load metadata DataFrame
        metadata_file = "nl_full_all_columns.csv"
        self.metadata_df = pd.read_csv(metadata_file).set_index("id")

        # Load embedding model
        self.model = SentenceTransformer(embedding_model_name)

    @staticmethod
    def normalize(vectors):
        """
        Normalize vectors for cosine similarity.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def query(self, query_text, top_k=5):
        """
        Query the FAISS index and retrieve the top-k results with metadata.

        Args:
            query_text (str): Query string to search for.
            top_k (int): Number of top results to retrieve.

        Returns:
            List[dict]: List of results with IDs, scores, and selected metadata.
        """
        # Step 1: Convert query text to a normalized vector
        query_vector = self.model.encode([query_text]).astype("float32")
        query_vector = self.normalize(query_vector)

        # Step 2: Search the FAISS index
        distances, indices = self.index.search(query_vector, top_k)

        # Step 3: Retrieve metadata for results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # No result
                continue
            result_id = self.metadata_df.index[idx]  # Retrieve ID from the DataFrame index
            metadata_row = self.metadata_df.loc[result_id]
            
            metadata = {
                "title": metadata_row["title"],
                "main_category": metadata_row["main_category"],
                "latitude": metadata_row["latitude"],
                "longitude": metadata_row["longitude"],
                "generated_text": metadata_row["generated_text"],
            }
            results.append({
                "id": result_id,
                "score": dist,
                "metadata": metadata,
            })
        return results
