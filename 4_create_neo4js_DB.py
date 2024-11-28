import json
from neo4j import GraphDatabase

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Function to add nodes and relationships to Neo4j
def add_to_neo4j(nodes_file, edges_file):
    # Load nodes and edges from JSON files
    with open(nodes_file, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(edges_file, "r", encoding="utf-8") as f:
        edges = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        # Add nodes in batch
        print("Adding nodes...")
        for name, description in nodes:
            session.run(
                "MERGE (n:Entity {name: $name, description: $description})",
                {"name": name, "description": description}
            )

        # Add relationships in batch
        print("Adding relationships...")
        for source, target, relationship in edges:
            session.run(
                """
                MATCH (a:Entity {name: $source_name}), (b:Entity {name: $target_name})
                MERGE (a)-[r:RELATIONSHIP {type: $type}]->(b)
                """,
                {"source_name": source, "target_name": target, "type": relationship}
            )

    driver.close()
    print("Data added to Neo4j successfully.")

# Main Execution
if __name__ == "__main__":
    nodes_file = "nodes.json"  # Input file for nodes
    edges_file = "edges.json"  # Input file for edges

    add_to_neo4j(nodes_file, edges_file)
