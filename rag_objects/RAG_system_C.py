import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time
import numpy as np

class RAGSystemC:
    def __init__(self, index_path, embedding_model_name, gemini_token_file):
        """
        Initialize the RAGSystemC with the retriever and LLM configuration.

        Args:
            index_path (str): Path to the FAISS index file.
            embedding_model_name (str): Name of the embedding model.
            gemini_token_file (str): Path to the file containing the Gemini API token.
        """
        # Initialize retriever
        self.index = faiss.read_index(index_path)
        metadata_file = "nl_full_all_columns.csv"
        self.metadata_df = pd.read_csv(metadata_file).set_index("id")
        self.model = SentenceTransformer(embedding_model_name)

        # Initialize LLM (Gemini)
        try:
            with open(gemini_token_file, 'r') as file:
                gemini_api_token = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The token file '{gemini_token_file}' was not found.")

        genai.configure(api_key=gemini_api_token)

        # Configuration for the summary generation step
        generation_config_summary = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1000,
            "response_mime_type": "text/plain",
        }
        system_prompt_summary = (
            "You are part of a RAG system in an app with audio guides for various locations. "
            "The user prompts for locations of certain type and a retriever gives information about them. "
            "Your task is to create a brief prose summary of the locations in the list, letting the user "
            "know what they can expect when following. You are given a list of locations with their "
            "metadata and part of their audioguide scripts."
        )
        self.llm_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config_summary,
            system_instruction=system_prompt_summary
        )

        # Configuration for the query enhancer step
        generation_config_enhancer = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 800,
            "response_mime_type": "text/plain",
        }
        system_prompt_enhancer = (
            "You are part of a simple RAG system in an app with audio guides for various locations. "
            "You will receive a user query for a list of locations. Your task is to add more related words "
            "to this query. The retriever is simple and could use more words for embeddings. "
            "Repeat the most important words (such as names or locations). For others use synonyms. "
            "Return only the enhanced query."
        )
        self.enhancer_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config_enhancer,
            system_instruction=system_prompt_enhancer
        )

        print("RAGSystemC initialized and ready.")

    @staticmethod
    def normalize(vectors):
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def query(self, user_query, max_text=150, top_k=8, summarize=False):
        """
        Query the system and retrieve a summarized response from the LLM.

        Args:
            user_query (str): The user's query.
            max_text (int): Maximum number of characters of the 'generated_text' to include per location.
            top_k (int): Number of top locations to retrieve.
            summarize (bool): Whether to call the LLM to generate a summary.

        Returns:
            dict: A dictionary containing:
                  - 'locations': List of location data.
                  - 'response': The LLM-generated summary (if summarize=True).
                  - 'query_enhancer_time': Time taken by the enhancer step.
                  - 'enhancer_response': The raw text from the enhancer step.
                  - 'enhancer_total_tokens': Total tokens used by the enhancer call.
                  - 'enhancer_prompt_tokens': Prompt tokens used by the enhancer call.
                  - 'query_embedding_time': Time taken to convert query to embedding.
                  - 'retrieval_time': Time taken to retrieve locations.
                  - 'summary_generation_time': Time taken to generate the summary (if summarize=True).
                  - 'summary_total_tokens': Total tokens used in the summary call (if summarize=True).
                  - 'summary_prompt_tokens': Tokens used for the summary prompt (if summarize=True).
        """
        # --------------------------------------------------------------------
        # STEP 1: Enhance the user query via Gemini
        # --------------------------------------------------------------------
        enhancer_start_time = time.perf_counter()
        enhancer_response = self.enhancer_model.generate_content(user_query)
        enhancer_time = time.perf_counter() - enhancer_start_time

        # Extract token usage for enhancer
        enhancer_total_tokens = enhancer_response.usage_metadata.total_token_count
        enhancer_prompt_tokens = enhancer_response.usage_metadata.prompt_token_count

        # Use the enhancer output as the new query
        enhanced_query = enhancer_response.text.strip()

        # --------------------------------------------------------------------
        # STEP 2: Convert the (enhanced) query to embedding
        # --------------------------------------------------------------------
        query_embedding_start_time = time.perf_counter()
        query_vector = self.model.encode([enhanced_query]).astype("float32")
        query_vector = self.normalize(query_vector)
        query_embedding_time = time.perf_counter() - query_embedding_start_time

        # --------------------------------------------------------------------
        # STEP 3: Retrieve top-k results
        # --------------------------------------------------------------------
        retrieval_start_time = time.perf_counter()
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # No result
                continue
            result_id = self.metadata_df.index[idx]
            metadata_row = self.metadata_df.loc[result_id]
            metadata = {
                "title": metadata_row["title"],
                "main_category": metadata_row["main_category"],
                "generated_text": metadata_row["generated_text"][:max_text],  # Limit text
                "latitude": metadata_row["latitude"],
                "longitude": metadata_row["longitude"],
            }
            results.append(metadata)
        retrieval_time = time.perf_counter() - retrieval_start_time

        # --------------------------------------------------------------------
        # STEP 4: Generate LLM summary (if requested)
        # --------------------------------------------------------------------
        if summarize:
            generation_start_time = time.perf_counter()
            context = "\n".join(
                [
                    f"Title: {item['title']}, Category: {item['main_category']}, Text: {item['generated_text']}"
                    for item in results
                ]
            )
            prompt = (
                f"User Query: {user_query}\n\n"  # show original user query in the summary
                f"Enhanced Query: {enhanced_query}\n\n"
                f"Locations:\n{context}\n\n"
                "Please summarize these locations for the user."
            )

            summary_response = self.llm_model.generate_content(prompt)
            generation_time = time.perf_counter() - generation_start_time

            # Extract token usage for summary
            total_tokens = summary_response.usage_metadata.total_token_count
            prompt_tokens = summary_response.usage_metadata.prompt_token_count
            llm_output_text = summary_response.text
        else:
            generation_time = 0
            total_tokens = 0
            prompt_tokens = 0
            llm_output_text = ""

        # --------------------------------------------------------------------
        # RETURN DICTIONARY
        # --------------------------------------------------------------------
        return {
            "locations": results,
            "response": llm_output_text,
            "query_enhancer_time": enhancer_time,
            "enhancer_response": enhanced_query,  
            "enhancer_total_tokens": enhancer_total_tokens,
            "enhancer_prompt_tokens": enhancer_prompt_tokens,
            "query_embedding_time": query_embedding_time,
            "retrieval_time": retrieval_time,
            "summary_generation_time": generation_time,
            "summary_total_tokens": total_tokens,
            "summary_prompt_tokens": prompt_tokens
        }
