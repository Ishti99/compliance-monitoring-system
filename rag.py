import numpy as np
from embeddings import cosine_similarity, titan_embedding

# search the dataframe for most relevant chunks
def search(df, query, top_n=80, min_score=0.25):
    query_vec = titan_embedding(query)
    scores = []
    for i, row in df.iterrows():
        score = cosine_similarity(query_vec, row["embedding"])
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores[:top_n]:
        if score >= min_score:
            chunk = df.iloc[idx]["text"]
            results.append({"text": chunk, "score": round(score, 4)})
    return results

# format retrieved chunks into a single context string for the llm
def build_context(results):
    context = ""
    for i, r in enumerate(results):
        context += f"Finding {i+1} (relevance: {r['score']}):\n{r['text']}\n\n"
    return context.strip()