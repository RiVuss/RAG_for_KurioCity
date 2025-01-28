Things to consider trying and I don't want to forget

- Summarization Quality Analysis
  If the audio script is effectively a summary, run an advanced summarization evaluation metric (e.g., ROUGE, BERTScore) against the Wikipedia text.
  Inspect how well the summarization conveys key topics, factual correctness, and style.

- Does AI-assigned importance align with textual prominence?

If a location is “important” according to the AI, do we see it mentioned frequently elsewhere? Are there more inbound references to it across the corpus?

Limitation:

- FAISS works only on GPU on Linux, not windows.
