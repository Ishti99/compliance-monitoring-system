import boto3
import json
import numpy as np
import pandas as pd
import spacy
import pickle
import os

nlp = spacy.load("en_core_web_md")

def get_bedrock_client():
    session = boto3.Session(profile_name="GSB570-BedrockOnly-490332585640")
    return session.client("bedrock-runtime", region_name="us-west-2")

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def titan_embedding(text):
    client = get_bedrock_client()
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    })
    response = client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response.get("body").read())
    return np.array(result.get("embedding"))

def chunk_text(text):
    chunks = []
    current_factory = ""
    current_chunk = []

    for line in text.split("\n"):
        if "Factory" in line and "Audit Report" in line:
            if current_chunk and current_factory:
                chunks.append(current_factory + "\n" + "\n".join(current_chunk))
            current_factory = line.strip()
            current_chunk = []
        elif line.strip().startswith("[") and line.strip().endswith("]"):
            if current_chunk and current_factory:
                chunks.append(current_factory + "\n" + "\n".join(current_chunk))
            current_chunk = [line.strip()]
        elif line.strip():
            current_chunk.append(line.strip())

    if current_chunk and current_factory:
        chunks.append(current_factory + "\n" + "\n".join(current_chunk))

    return chunks

def build_dataframe(filepath):
    # check if we already embedded this file before
    cache_path = filepath + ".pkl"
    if os.path.exists(cache_path):
        print(f"found cached embeddings, loading from {cache_path}...")
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"done - {len(df)} chunks loaded")
        return df

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("chunking text...")
    chunks = chunk_text(text)
    print(f"embedding {len(chunks)} chunks using titan...")

    rows = []
    for chunk in chunks:
        vec = titan_embedding(chunk)
        rows.append({"text": chunk, "embedding": vec})

    df = pd.DataFrame(rows)

    # save so we don't have to redo this next time
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"done - {len(df)} chunks ready")

    return df