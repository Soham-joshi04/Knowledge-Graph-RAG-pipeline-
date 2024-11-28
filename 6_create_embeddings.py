import json
import requests

# Embedding API URL
API_URL = "http://localhost:11434/api/embeddings"

# Function to generate embeddings using the provided API
def generate_embedding_custom_api(text):
    data = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding:
            return embedding
        else:
            print(f"Warning: 'embedding' key missing in response for text: {text[:30]}...")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error generating embedding for text: {text[:30]}... -> {e}")
        return None

# Function to process nodes, relationships, and summaries
def generate_indexed_embeddings(nodes_file, summaries_file, output_file):
    with open(nodes_file, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    # with open(relationships_file, "r", encoding="utf-8") as f:
    #     relationships = json.load(f)

    with open(summaries_file, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    indexed_embeddings = {
        "nodes": [],
        "relationships": [],
        "summaries": []
    }

    # Process nodes
    print("Processing nodes...")
    for idx, (name, description) in enumerate(nodes):
        text = f"{name}: {description}"
        embedding = generate_embedding_custom_api(text)
        if embedding:
            print(f"{name}: embedded")
            indexed_embeddings["nodes"].append({
                "id": f"node_{idx}",
                "name": name,
                "description": description,
                "embedding": embedding
            })

    # Process relationships
    # print("Processing relationships...")
    # for idx, (source, target, relationship) in enumerate(relationships):
    #     text = f"{source} -[{relationship}]-> {target}"
    #     embedding = generate_embedding_custom_api(text)
    #     if embedding:
    #         indexed_embeddings["relationships"].append({
    #             "id": f"relationship_{idx}",
    #             "source": source,
    #             "target": target,
    #             "relationship": relationship,
    #             "embedding": embedding
    #         })

    # Process summaries
    print("Processing summaries...")
    for idx, (file_name, summary) in enumerate(summaries.items()):
        embedding = generate_embedding_custom_api(summary)
        if embedding:
            print(f"{file_name}: embedded")
            indexed_embeddings["summaries"].append({
                "id": f"summary_{idx}",
                "file_name": file_name,
                "summary": summary,
                "embedding": embedding
            })

    # Save indexed embeddings to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(indexed_embeddings, f, indent=4)

    print(f"Indexed embeddings saved to {output_file}")

# Main Execution
if __name__ == "__main__":
    # File paths
    nodes_file = "indexes/nodes.json"
    # relationships_file = "edges.json"
    summaries_file = "indexes/file_summaries.json"
    output_file = "indexes/indexed_embeddings.json"

    generate_indexed_embeddings(nodes_file, summaries_file, output_file)
