import json
import requests
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from neo4j import GraphDatabase

# API Configuration
EMBEDDING_API_URL = "http://localhost:11434/api/embeddings"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
LLM_MODEL = "llama3.2"

# Initialize LLM and Neo4j Driver
llm = Ollama(model=LLM_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function to generate embedding for a query
def generate_query_embedding(query):
    payload = {
        "model": "nomic-embed-text",
        "prompt": query
    }
    response = requests.post(EMBEDDING_API_URL, json=payload)
    response.raise_for_status()
    embedding = response.json().get("embedding")
    if not embedding:
        raise ValueError(f"Failed to retrieve embedding for query: {query}")
    return np.array(embedding)

# Function to analyze query and determine type
def analyze_query(query):
    prompt = PromptTemplate(
        template="""
        Query: {query}
        Task: Determine the query type:
        - Is it asking about relationship between two nodes? If yes, specify the two node names.
        - Is it global (file summaries required, a query which does not ask about specifics) or local (nodes required,eg query asking about how does a pawn moves)? 
        Also, indicate how many nodes or file summaries are required.
        
        Format: 
        Relationship: <yes/no>, Node1: <name>, Node2: <name>
        Global: <yes/no>, Local: <yes/no>, Nodes: <number>, Files: <number>
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"query": query}).replace("\n", "").strip()

    # Normalize the response
    response = response.replace("Global:", ", Global:")
    response = response.replace("Local:", ", Local:")
    response = response.replace("Nodes:", ", Nodes:")
    response = response.replace("Files:", ", Files:")
    response = response.replace("Node1:", ", Node1:")
    response = response.replace("Node2:", ", Node2:")

    try:
        parts = response.split(", ")
        relationship = "yes" in parts[0].lower()
        node1 = parts[1].split(":")[1].strip() if relationship else None
        node2 = parts[2].split(":")[1].strip() if relationship else None
        global_ = "yes" in parts[3].lower()
        local = "yes" in parts[4].lower()
        num_nodes = int(parts[5].split(":")[1].strip()) if "Nodes" in parts[5] else 1
        num_files = int(parts[6].split(":")[1].strip()) if "Files" in parts[6] else 1
        return relationship, node1, node2, global_, local, num_nodes, min(num_files, 1)  # Limit to 1 file
    except Exception as e:
        print(f"Error parsing LLM response: {response}, Error: {e}")
        return False, None, None, True, True, 1, 1


# Function to retrieve relationships and descriptions for two nodes
def retrieve_relationship_between_nodes(node1, node2):
    with driver.session() as session:
        query = """
        MATCH (a:Entity {name: $node1})
        OPTIONAL MATCH (a)-[r]->(b:Entity {name: $node2})
        OPTIONAL MATCH (b)-[r2]->(a)
        RETURN 
            a.name AS node1_name, a.description AS node1_description,
            b.name AS node2_name, b.description AS node2_description,
            r.type AS relationship1, r2.type AS relationship2
        """
        results = session.run(query, {"node1": node1, "node2": node2})
        context = f"Node 1: {node1}\n"
        context += f"Node 2: {node2}\n"
        relationship_found = False
        for record in results:
            context += f"Node 1 Description: {record['node1_description']}\n"
            context += f"Node 2 Description: {record['node2_description']}\n"
            if record["relationship1"]:
                context += f"Relationship: {node1} -[{record['relationship1']}]-> {node2}\n"
                relationship_found = True
            if record["relationship2"]:
                context += f"Relationship: {node2} -[{record['relationship2']}]-> {node1}\n"
                relationship_found = True
        if not relationship_found:
            context += "No direct relationship exists between the two nodes.\n"
        return context

# Function to retrieve top N similar embeddings
def retrieve_similar_embeddings(query_embedding, embeddings, top_n):
    similarities = [
        (idx, cosine_similarity([query_embedding], [np.array(e["embedding"])]).item())
        for idx, e in enumerate(embeddings)
    ]
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return [(embeddings[idx], sim) for idx, sim in sorted_similarities]

# Function to retrieve node context from Neo4j
def retrieve_node_context_from_neo4j(node_name):
    with driver.session() as session:
        query = """
        MATCH (n:Entity {name: $name})
        OPTIONAL MATCH (n)-[r]->(m)
        OPTIONAL MATCH (m)-[r2]->(n)
        RETURN 
            n.name AS source_name, n.description AS source_description,
            r.type AS relationship, 
            m.name AS target_name, m.description AS target_description
        """
        results = session.run(query, {"name": node_name})
        context = f"Node: {node_name}\n"
        for record in results:
            if record["relationship"] and record["target_name"]:
                context += f"{node_name} -[{record['relationship']}]-> {record['target_name']} ({record['target_description']})\n"
        return context

# Function to build final context
def build_final_context(query, embeddings, summaries):
    # Analyze the query
    relationship, node1, node2, global_, local, num_nodes, num_files = analyze_query(query)

    if relationship and node1 and node2:
        return retrieve_relationship_between_nodes(node1, node2)

    query_embedding = generate_query_embedding(query)
    context = ""

    if local:
        similar_nodes = retrieve_similar_embeddings(query_embedding, embeddings["nodes"], num_nodes)
        for node, sim in similar_nodes:
            node_context = retrieve_node_context_from_neo4j(node["name"])
            context += f"Similarity: {sim:.2f}\n{node_context}\n"

    if global_:
        similar_files = retrieve_similar_embeddings(query_embedding, embeddings["summaries"], num_files)
        for file, sim in similar_files:
            context += f"Similarity: {sim:.2f}\nFile: {file['file_name']}\nSummary: {file['summary']}\n\n"

    return context.strip()

# Function to get final LLM response
def get_response(query, context):
    prompt = PromptTemplate(
        template="""
        Query: {query}
        Context: {context}

        Task: Answer the query based on the given context. 
        Note: The context is arranged in decreasing order of relevance.
        If query is not matching with given context then reply with not enoughh context.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"query": query, "context": context})

# Main function for query pipeline
def query_pipeline(query, embeddings_file, summaries_file):
    with open(embeddings_file, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    with open(summaries_file, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    # Build final context
    context = build_final_context(query, embeddings, summaries)

    # Get final response
    response = get_response(query, context)
    return response

# Main execution
if __name__ == "__main__":
    query = input("Enter your query: ")
    embeddings_file = "indexes/indexed_embeddings.json"
    summaries_file = "indexes/file_summaries.json"

    response = query_pipeline(query, embeddings_file, summaries_file)
    print("\nResponse:")
    print(response)
