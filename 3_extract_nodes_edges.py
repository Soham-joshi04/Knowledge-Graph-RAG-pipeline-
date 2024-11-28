import json
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure Llama 3.2
llm = Ollama(model="llama3.2")

# Relationship Extraction Prompt Template
relationship_extraction_prompt = PromptTemplate(
    template="""
    Context: {context}

    Entity 1: {source} ({source_description})
    Entity 2: {target} ({target_description})

    Task: Identify the relationship between these two entities based on the provided context. 
    If no relationship exists, respond with "No relationship."
    """
)

# Function to extract relationship using LLM
def extract_relationship(context, source, source_description, target, target_description):
    chain = LLMChain(llm=llm, prompt=relationship_extraction_prompt)
    response = chain.run({
        "context": context,
        "source": source,
        "source_description": source_description,
        "target": target,
        "target_description": target_description
    })

    if "No relationship" in response:
        return None
    return response.strip()

# Function to process entities and relationships for all chunks
def process_entities_and_relationships(entities_file, chunks_file, output_nodes_file, output_edges_file):
    # Load entities and chunk data
    with open(entities_file, "r", encoding="utf-8") as f:
        entities_data = json.load(f)

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    nodes = set()  # Store unique nodes as (entity_name, description)
    edges = []     # Store edges as (source, target, relationship)

    for file_name, chunks in entities_data.items():
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_context = chunks_data[file_name][chunk_id]["text"]

            print(f"Processing {file_name}: Chunk {chunk_id}...")

            for entity in chunk["entities"]:
                source_name = entity["entity"]
                source_description = entity["description"]

                # Add source node
                nodes.add((source_name, source_description))

                # Process related entities
                if "relations" in entity and entity["relations"]:
                    for target_name in entity["relations"]:
                        # Get target description from entities in the chunk
                        target_description = next(
                            (e["description"] for e in chunk["entities"] if e["entity"] == target_name),
                            "No description"
                        )

                        # Extract relationship using LLM
                        relationship = extract_relationship(
                            chunk_context, source_name, source_description, target_name, target_description
                        )

                        # Skip if no relationship is found
                        if not relationship:
                            continue

                        # Add target node and edge
                        nodes.add((target_name, target_description))
                        edges.append((source_name, target_name, relationship))

    # Save nodes to JSON
    with open(output_nodes_file, "w", encoding="utf-8") as f:
        json.dump(list(nodes), f, indent=4)

    # Save edges to JSON
    with open(output_edges_file, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=4)

    print(f"Nodes saved to {output_nodes_file}")
    print(f"Edges saved to {output_edges_file}")

# Main Execution
if __name__ == "__main__":
    entities_file = "indexes/extracted_entities.json"  # File with extracted entities and relations
    chunks_file = "indexes/output_chunks.json"         # File with chunk texts
    output_nodes_file = "indexes/nodes.json"           # Output file for nodes
    output_edges_file = "indexes/edges.json"           # Output file for edges

    process_entities_and_relationships(entities_file, chunks_file, output_nodes_file, output_edges_file)
