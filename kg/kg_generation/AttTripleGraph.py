import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import json
#  "context": {
#       "preceding_text": "The paper discusses how",
#     "following_text": "This distinction is crucial for AI applications."
#  },
triplets = [
    {
        "type": "semantic",
        "subject": "Unifying Large Language Models and Knowledge Graphs: A Roadmap",
        "predicate": "contains_statement",
        "object": "LLMs capture vast amounts of implicit knowledge, while KGs store explicit structured information.",
        "attributes": {
            "source": "highlight",
            "position": "45-140",
            "context": "The paper discusses how ... This distinction is crucial for AI applications.",
            "document_id": "document_uri_1",
            "timestamp": "2025-04-01T09:15:00Z"
        }
    },
    {
        "type": "user_intent",
        "subject": "username1",
        "predicate": "indicates_importance_of",
        "object": "LLMs capture vast amounts of implicit knowledge, while KGs store explicit structured information.",
        "attributes": {
            "comment": "This sentence summarizes the core advantage of integrating LLMs with KGs.",
            "source": "comment",
            "document_id": "document_uri_1",
            "timestamp": "2025-04-01T09:15:00Z",
            "tags": ["LLM", "KG", "integration", "implicit knowledge"]
        }
    },
    {
        "type": "semantic",
        "subject": "Unifying Large Language Models and Knowledge Graphs: A Roadmap",
        "predicate": "details_frameworks",
        "object": "The roadmap outlines three frameworks: KG-enhanced LLMs, LLM-augmented KGs, and synergized LLMs+KGs.",
        "attributes": {
            "source": "highlight",
            "position": "150-230",
            "context": "Specifically, ... Each framework offers distinct benefits for knowledge integration.",
            "document_id": "document_uri_1",
            "timestamp": "2025-04-01T09:18:00Z"
        }
    },
    {
        "type": "user_intent",
        "subject": "username1",
        "predicate": "expresses_interest_in",
        "object": "framework classification",
        "attributes": {
            "comment": "This classification of frameworks highlights different strategies for knowledge integration.",
            "source": "comment",
            "document_id": "document_uri_1",
            "timestamp": "2025-04-01T09:18:00Z",
            "tags": ["framework", "classification", "integration"]
        }
    },
    {
        "type": "semantic",
        "subject": "KGAREVION: An AI Agent for Knowledge-Intensive Biomedical QA",
        "predicate": "describes_pipeline",
        "object": "KGAREVION utilizes a generate-review-revise pipeline to construct biomedical knowledge graphs.",
        "attributes": {
            "source": "highlight",
            "position": "30-110",
            "context": "The system is designed so that ... ensuring high quality and accurate representation.",
            "document_id": "document_uri_2",
            "timestamp": "2025-04-01T10:05:00Z"
        }
    },
    {
        "type": "user_intent",
        "subject": "username1",
        "predicate": "notes_method",
        "object": "generate-review-revise pipeline",
        "attributes": {
            "comment": "This method inspires our approach for validating and refining attributed triples.",
            "source": "comment",
            "document_id": "document_uri_2",
            "timestamp": "2025-04-01T10:05:00Z",
            "tags": ["generate-review-revise", "pipeline", "biomedical"]
        }
    },
    {
        "type": "semantic",
        "subject": "KGAREVION: An AI Agent for Knowledge-Intensive Biomedical QA",
        "predicate": "demonstrates_extraction",
        "object": "The agent leverages a fine-tuned LLM to extract high-quality triples from biomedical texts.",
        "attributes": {
            "source": "highlight",
            "position": "120-210",
            "context": "In its implementation, ... which significantly reduces errors.",
            "document_id": "document_uri_2",
            "timestamp": "2025-04-01T10:08:00Z"
        }
    },
    {
        "type": "user_intent",
        "subject": "username1",
        "predicate": "recognizes_technique",
        "object": "fine-tuning LLM for precise triple extraction",
        "attributes": {
            "comment": "The core technology lies in fine-tuning the LLM to generate precise knowledge representations.",
            "source": "comment",
            "document_id": "document_uri_2",
            "timestamp": "2025-04-01T10:08:00Z",
            "tags": ["fine-tuning", "LLM", "quality", "triples"]
        }
    }
]

G = nx.DiGraph()

def add_triple_to_graph(triple):
    subj = triple["subject"]
    pred = triple["predicate"]
    obj = triple["object"]
    attrs = triple["attributes"]
    triple_type = triple["type"]
    if not G.has_node(subj):
        G.add_node(subj)
    if not G.has_node(obj):
        G.add_node(obj)
 
    G.add_edge(subj, obj, predicate=pred, triple_type=triple_type, **attrs)

for triple in triplets:
    add_triple_to_graph(triple)

print("Nodes in the Knowledge Graph:")
for node in G.nodes():
    print(node)

print("\nEdges in the Knowledge Graph:")
for u, v, data in G.edges(data=True):
    print(f"{u} --[{data['predicate']}, type: {data['triple_type']}]--> {v}, Attributes: { {k: v for k, v in data.items() if k not in ['predicate', 'triple_type']} }")

data = json_graph.node_link_data(G)

# 将数据写入 JSON 文件
with open("kg.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Knowledge Graph saved to kg.json")

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', arrowsize=20, node_size=1500)
edge_labels = nx.get_edge_attributes(G, 'predicate')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
plt.title("Knowledge Graph Constructed from Synthetic Attributed Triplets with Type")
plt.show()
