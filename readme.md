# [KurioCity](https://kuriocity.com)   Audio Guide Route Generation

 <img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/logo_kurioCity.png" width="50%" />

## 📖 Background

KurioCity is an app that provides **audio guides** for everywhere in the Netherlands. However, with an **abundance of locations**, it becomes difficult for users to find the best places that match their interests.

### 🎯 The Solution:
In this repo, I'm experimenting with **AI-powered custom route generation** using **Retrieval-Augmented Generation (RAG)** to create personalized tourist routes based on their queries/preferences. Such system could make it much easier for app users to decide which locations to check out!

#### Example Input & Output:
#### **Input:**  
_"What should a Rembrandt lover see in Leiden?"_
#### **Output:**  
_List of relevant locations + AI-generated summary._

## 📊 Dataset summary

The dataset of KurioCity audio guides in the Netherlands covers **40,107 locations**. 

Data columns: **Title**, **Wikipedia article**, **Audio guide script**, **Categories & subcategories**, **Importance ranking**, **Thumbnail images**.

Cool stats about it are below! 

## 🏗️ RAG System Designs

To find out what would such a system need, I experimented with **three different RAG architectures**.

### 🔹 **System A** (Basic)
1. User query  
2. **Retriever**: Fetches a **list of locations** (8) from the index. 
3. **AI Summarizer**: Generates a route summary based on the selected locations. 

### 🔹 **System B** (With AI Selector)
1. User query  
2. **Retriever**: Fetches a **longer** list of locations (16) from the index.
3. **AI Selector**: Picks 8 **best** locations for the query.
4. **AI Summarizer**: Generates a route summary based on the selected locations.

### 🔹 **System C** (With Query Enhancer)
1. User query  
2. **AI Query Enhancer**: Gets the user query and enhances it with additional related words to make the job easier for the retriever.
3. **Retriever**: Fetches a **list of locations** (8) from the index.  
4. **AI Summarizer**: Generates a route summary based on the selected locations.

## 🧠 Model Selection

I tried 3 **embedding models** for the retrieval:

| Model | Parameters | Embedding Size |
|--------|------------|----------------|
| `all-MiniLM-L6-v2` | 22.7M | 256 |
| `all-mpnet-base-v2` | 109M | 768 |
| `NovaSearch stella_en_1.5B_v5` | 1.5B | 1024 |

The idea was to try different model and embedding sizes.
Across the experiments, the best model was **all-mpnet-base-v2**, however **NovaSearch stella_en_1.5B_v5** showed potential in a use case with longer queries.

For indexing, I tried **Qdrant** and **Facebook AI Similarity Search (FAISS)**. In the end I used FAISS, as I found it more intuitive and faster. 

**Tested LLMs:**

For summaries, query enhancement and location selection, an LLM was also needed. I experimented with
- **Yi-6b-200K** (Local, long context, poor output)
- **Gemma-2-2b** (Local, short context, poor output)
- **Gemini 1.5 Flash** (API-based, long context, best results)
Ended up using Gemini, as the models I could run on my PC were too slow and inaccurate.

## ⚡ Metrics

**3 RAG architectures x 3 embedding models** = **9 experiments**.

8 test queries, 5 ideal locations for each (golden route)

### ✅ Evaluation Metrics:
- **Golden Route Hit Rate**
- **Time Taken**
- **Tokens Used**
- **Qualitative assessment** - each retrieved location was rated manually on a scale: good, relevant, irrelevant, wrong area.

## 📈 Results

- **Best Performing System**:  
  ✅ **AI Selector + Medium Embedding Model**  
  ✅ **45% "good" locations found**  
  ✅ **65% "okay" locations found**
 <img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/good_locations.png" width="80%" />


- Overall, the system shows potential, yet it is far from being consumer-ready. For now, it will be used as an internal tool to assist in route-making.

- **AI Checker Helped**, but **not always**
- **Query Enhancer was beneficial**, but sometimes **retrieved results from the wrong area**  
<img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/peformance_breakdown_per_experiment.png" width="80%" />


### ⏳ Retrieval Speed:
- **Retrieval time**: Almost negligible
- **Embedding query time**: Increases with model size, still very short
- **API response time**: Sub-second, but longest overall

<img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/time_breakdowns.png" width="70%" />


### 🔥 Best Queries:
- _"What to see in Amsterdam to learn about the Jewish heritage?"_ (33/72 good)
- _"What should a museum nerd see in Haarlem?"_ (31/72 good)

### ❌ Worst Queries:
- _"I want to see the most famous bridges of Amsterdam"_ (9/72 good)
- _"What should a Rembrandt lover see in Leiden?"_ (20/72 good)


## 🔮 Future Improvements

Though this system is unlikely to reach the app soon, it will be very useful for my internal work of making audio guide routes. To make it work even better, I'm considering these enhancements
- **📌Improve Location Accuracy**: **Named Entity Recognition + Geohashes** to limit the area of the search, giving more relevant locations
- **📑 Chunked Embeddings**: For more accurate retrieval, especially in longer audio guides
- **🧠 Test More Embedding Models**: One big conclusion is that size is not everything when it comes to the embeddings. It could be worth it to explore more models to find the best one. 
- **🔄 Combine AI Checker + Query Enhancer**

## 📊 More on the dataset

As a fun side analysis, I applied **Latent Dirichlet Allocation (LDA)** on the dataset to check out any interesting geo-realations of topics.

<img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/word_clouds.png" width="60%" />

<div style="display: flex; width: 100%;">
  <img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/public_transport.png" style="width: 33.33%; height: 300px; object-fit: cover;"/>
  <img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/nobelty_and_estates.png" style="width: 33.33%; height: 300px; object-fit: cover;"/> 
  <img src="https://github.com/RiVuss/RAG_for_KurioCity/blob/main/visualizations/churches_and_religion.png" style="width: 33.33%; height: 300px; object-fit: cover;"/>
</div>

All the maps can be found in code, but here are some cool observations:
- the public transport neatly follows rail routes
- estates are more gathered by the sea (who does not like a sea property)
- religion-related locations are more dense in the South, which correlates with the spread of religiousness within the Netherlands.


## **🔗 Website:** [KurioCity.com](https://kuriocity.com)  

