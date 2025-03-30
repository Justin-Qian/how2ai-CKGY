import json
from py2neo import Graph
from extractors.triplet_extractor import extract_triplets
from storage.neo4j_uploader import upload_to_neo4j

# Load JSON file
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    data = load_json("data/annotations.json")
    triplets = extract_triplets(data)

    # 🔐 Connect to Neo4j (update password if you changed it)
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "H2AI2025"))
    
    upload_to_neo4j(triplets, graph)

if __name__ == "__main__":
    main()
