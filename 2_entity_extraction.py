import json
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure Llama 3.2
llm = Ollama(model="llama3.2")

# Entity Extraction Prompt Template
entity_extraction_prompt = PromptTemplate(
    template="""
    Context: {context}

    Task: Extract all unique entities and their relationships from the provided context. 
    For each entity, include its description and a list of related entities for a given entity in relations. For example out of total 10 entity in the context entity 1 is related to entities 2, 3 only then relations should have ["Entity2", "Entity3"]. Note if entity 2 is in relation of entity 1 then entity 1 should be strictly in relation of entity 2. Also if entity is present in the context but not related to any other entity then relations should be an empty list. Strictly avoid duplicate entities. Also
    Respond in JSON format only:
    [
        {{ "entity": "EntityName1", "description": "Description of EntityName1", "relations": [ "EntityName2", "EntityName3" ]}},
        {{ "entity": "EntityName2", "description": "Description of EntityName2", "relations": ["EntityName1"] }}
    ]
    Strictly do not include duplicate entities in this list. Keep response in JSON format only as mentioned above. no need to give any additional explaination for your output.
    """
)

# Function to extract entities and relationships from a single chunk
def extract_entities_for_chunk(context):
    chain = LLMChain(llm=llm, prompt=entity_extraction_prompt)
    response = chain.run({"context": context})

    try:
        entities_with_relations = json.loads(response)
    except json.JSONDecodeError:
        print("Error parsing LLM response.")
        return []

    return entities_with_relations

# Function to process all chunks in a JSON file
def extract_entities_from_chunks(chunks_file, output_file):
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    extracted_data = {}

    for file_name, chunks in chunks_data.items():
        extracted_data[file_name] = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_context = chunk["text"]

            print(f"Processing {file_name}: Chunk {chunk_id}...")

            # Extract entities for the chunk
            entities_with_relations = extract_entities_for_chunk(chunk_context)

            # Store the result with the chunk ID
            extracted_data[file_name].append({
                "chunk_id": chunk_id,
                "entities": entities_with_relations
            })

    # Save the extracted entities to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Entity extraction completed. Results saved to {output_file}.")

# Main execution
if __name__ == "__main__":
    chunks_file = "indexes/output_chunks.json"       # Input file with chunk texts
    output_file = "indexes/extracted_entities.json" # Output file to save extracted entities

    extract_entities_from_chunks(chunks_file, output_file)
