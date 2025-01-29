import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time
import numpy as np

class RAGSystemB:
    def __init__(self, 
                 index_path, 
                 embedding_model_name, 
                 gemini_token_file,
                 checker_model_name="gemini-1.5-flash"):
        """
        Initialize the RAGSystemB with the retriever and LLM configuration.
        """
        # Initialize retriever
        self.index = faiss.read_index(index_path)
        metadata_file = "nl_full_all_columns.csv"
        self.metadata_df = pd.read_csv(metadata_file).set_index("id")
        self.model = SentenceTransformer(embedding_model_name)

        # Load Gemini API token
        try:
            with open(gemini_token_file, 'r') as file:
                gemini_api_token = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The token file '{gemini_token_file}' was not found.")

        genai.configure(api_key=gemini_api_token)

        # LLM for Summaries
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1000,
            "response_mime_type": "text/plain",
        }

        system_prompt_summary = (
            "You are part of a RAG system in an app with audio guides for various locations. "
            "The user prompts for locations of certain type and a retriever gives information "
            "about them. Your task is to create a brief prose summary of the locations in the "
            "list, letting the user know what they can expect."
        )
        self.llm_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=system_prompt_summary
        )

        # LLM for Checker
        checker_system_prompt = (
            "You are part of a RAG system in an app with audio guides for various locations. "
            "You will receive the user query and a list of retrieved locations. Your task is to "
            "select 8 locations best matching the query. Output them as a simple list of "
            "titles. E.g. [title1, title2, title3]."
        )
        self.llm_checker = genai.GenerativeModel(
            model_name=checker_model_name,
            generation_config=generation_config,
            system_instruction=checker_system_prompt
        )

        print("RAGSystemB initialized and ready.")

    @staticmethod
    def normalize(vectors):
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def query(self, user_query, max_text=150, top_k=5, summarize=True):
        """
        Query the system and retrieve a summarized response from the LLM.
        """
        # Step 1: Convert query to embedding
        query_embedding_start_time = time.perf_counter()
        query_vector = self.model.encode([user_query]).astype("float32")
        query_vector = self.normalize(query_vector)
        query_embedding_time = time.perf_counter() - query_embedding_start_time

        # Step 2: Retrieve top-k results
        retrieval_start_time = time.perf_counter()
        distances, indices = self.index.search(query_vector, top_k)
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 if no result
                continue
            result_id = self.metadata_df.index[idx]
            metadata_row = self.metadata_df.loc[result_id]
            metadata = {
                "title": metadata_row["title"],
                "main_category": metadata_row["main_category"],
                "generated_text": metadata_row["generated_text"][:max_text],
                "latitude": metadata_row["latitude"],
                "longitude": metadata_row["longitude"],
            }
            results.append(metadata)
        retrieval_time = time.perf_counter() - retrieval_start_time

        # Step 3: Checker step to filter results
        checker_start_time = time.perf_counter()

        # Convert retrieved locations into a readable format for the checker LLM
        locations_as_str = "\n".join(
            [f"- {r['title']} ({r['main_category']}): {r['generated_text']}" for r in results]
        )

        checker_prompt = (
            f'User query: "{user_query}"\n'
            f"Locations:\n{locations_as_str}\n\n"
        )
        print(checker_prompt)
        checker_response_obj = self.llm_checker.generate_content(checker_prompt)
        checker_time = time.perf_counter() - checker_start_time

        checker_response_text = checker_response_obj.text.lower()  # Convert to lowercase for robust matching

        # AI-checker token usage
        checker_total_tokens = checker_response_obj.usage_metadata.total_token_count
        checker_prompt_tokens = checker_response_obj.usage_metadata.prompt_token_count

        # Filter retrieved locations based on substring matching
        filtered_results = [item for item in results if item["title"].lower() in checker_response_text]

        # Step 4: Generate final summary (optional)
        if summarize and filtered_results:
            generation_start_time = time.perf_counter()

            context = "\n".join(
                [f"Title: {item['title']}, Category: {item['main_category']}, Text: {item['generated_text']}"
                 for item in filtered_results]
            )
            final_prompt = (
                f"User Query: {user_query}\n\n"
                f"Locations:\n{context}\n\n"
                f"Please summarize these locations for the user."
            )

            response_obj = self.llm_model.generate_content(final_prompt)
            llm_out = response_obj.text

            summary_generation_time = time.perf_counter() - generation_start_time

            # Summary usage
            summary_total_tokens = response_obj.usage_metadata.total_token_count
            summary_prompt_tokens = response_obj.usage_metadata.prompt_token_count

        else:
            # Either summarization is turned off or no results to summarize
            llm_out = ""
            summary_generation_time = 0
            summary_total_tokens = 0
            summary_prompt_tokens = 0

        # Step 5: Return results
        return {
            "locations": filtered_results,  # The final list after checker
            "response": llm_out,
            "checker_response": checker_response_text,
            "query_embedding_time": query_embedding_time,
            "retrieval_time": retrieval_time,
            "checker_time": checker_time,
            "summary_generation_time": summary_generation_time,
            "summary_total_tokens": summary_total_tokens,
            "summary_prompt_tokens": summary_prompt_tokens,
            "checker_total_tokens": checker_total_tokens,
            "checker_prompt_tokens": checker_prompt_tokens
        }
