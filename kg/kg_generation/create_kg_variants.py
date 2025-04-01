import networkx as nx
import json
from networkx.readwrite import json_graph

def load_kg(filename="kg.json"):
    """Load knowledge graph from JSON file"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create a new DiGraph
        G = nx.DiGraph()
        
        # Add nodes from the node-link format
        for node in data["nodes"]:
            G.add_node(node["id"])
        
        # Add edges from the node-link format
        for link in data["links"]:
            source = link["source"]
            target = link["target"]
            # Extract predicate and other attributes
            predicate = link.get("predicate", "")
            triple_type = link.get("triple_type", "semantic")
            # Remove these keys to avoid duplication as they'll be specified separately
            attrs = {k: v for k, v in link.items() 
                    if k not in ["source", "target", "predicate", "triple_type"]}
            
            # Add the edge with all attributes
            G.add_edge(source, target, predicate=predicate, triple_type=triple_type, **attrs)
            
        return G
    except Exception as e:
        print("Error loading KG:", e)
        return None

def create_kg_no_annotations(input_kg="kg.json", output_kg="kg_no_annotations.json"):
    """Create a KG variant without user annotation triples"""
    
    # Load the original KG
    G = load_kg(input_kg)
    
    if G is None:
        print("Failed to load original KG")
        return
    
    # Create a new graph containing only semantic triples (no user_intent)
    G_no_annotations = nx.DiGraph()
    
    # Copy all nodes
    for node in G.nodes():
        G_no_annotations.add_node(node)
    
    # Copy only semantic edges (no user_intent)
    for u, v, data in G.edges(data=True):
        if data.get("triple_type") == "semantic":
            G_no_annotations.add_edge(u, v, **data)
    
    # Convert to node-link format
    data = json_graph.node_link_data(G_no_annotations)
    
    # Save to file
    with open(output_kg, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Created knowledge graph without annotations: {output_kg}")
    print(f"Original graph had {len(G.edges())} edges, new graph has {len(G_no_annotations.edges())} edges")

def create_ablation_variants():
    """Create KG variants for ablation studies"""
    
    G = load_kg("kg.json")
    
    if G is None:
        print("Failed to load original KG")
        return
    
    # Variant 1: Without User Intent Triples
    G_no_user_intent = nx.DiGraph()
    for node in G.nodes():
        G_no_user_intent.add_node(node)
    for u, v, data in G.edges(data=True):
        if data.get("triple_type") == "semantic":
            G_no_user_intent.add_edge(u, v, **data)
    data = json_graph.node_link_data(G_no_user_intent)
    with open("kg_no_user_intent.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # Variant 2: Without Semantic Triples
    G_no_semantic = nx.DiGraph()
    for node in G.nodes():
        G_no_semantic.add_node(node)
    for u, v, data in G.edges(data=True):
        if data.get("triple_type") == "user_intent":
            G_no_semantic.add_edge(u, v, **data)
    data = json_graph.node_link_data(G_no_semantic)
    with open("kg_no_semantic.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # Variant 3: Without Attribute Context
    G_no_attributes = nx.DiGraph()
    for node in G.nodes():
        G_no_attributes.add_node(node)
    for u, v, data in G.edges(data=True):
        stripped_data = {
            "predicate": data.get("predicate", ""),
            "triple_type": data.get("triple_type", "semantic")
        }
        G_no_attributes.add_edge(u, v, **stripped_data)
    data = json_graph.node_link_data(G_no_attributes)
    with open("kg_no_attributes.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("Created all ablation variants of the knowledge graph")

if __name__ == "__main__":
    # Create KG variant without annotations for baseline C
    create_kg_no_annotations()
    
    # Create variants for ablation studies
    create_ablation_variants() 