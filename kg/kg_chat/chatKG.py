import networkx as nx
import matplotlib.pyplot as plt
import openai
import re
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# If API key is not found, print a warning
if not openai.api_key:
    print("WARNING: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

def load_kg(filename="evaluation/evaluation_data/kg.json"):
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

G = load_kg()
if G is None:
    print("No KG loaded. Exiting.")
    exit(1)

# 2. 检索相关三元组函数
def retrieve_relevant_triples(question, graph, top_k=5):
    """
    Keywords Matching, make KG's every edge's subject, predicate, object
    """
    # parse questions
    question_words = set(re.sub(r'[^a-zA-Z0-9 ]', '', question).lower().split())
    relevant_triples = []
    
    for u, v, data in graph.edges(data=True):
        triple_text = f"{u} {data['predicate']} {v}".lower()
        if any(word in triple_text for word in question_words):
            relevant_triples.append((u, data['predicate'], v, data))
    
    # 返回前 top_k 个匹配结果
    return relevant_triples[:top_k]

# 3. 构造增强提示函数
def construct_prompt(question, retrieved_triples):
    prompt = f"User's question: {question}\n\n"
    prompt += "Relevant facts from your notes:\n"
    for (subj, pred, obj, data) in retrieved_triples:
        # 构造描述时，可以根据三元组类型进行差异化展示
        if data.get("triple_type") == "semantic":
            prompt += f"- From the document '{data.get('document_id', '')}': {subj} {pred} {obj}.\n"
        elif data.get("triple_type") == "user_intent":
            # 对于用户意图三元组，可以加入评论信息
            comment = data.get("comment", "")
            prompt += f"- Note: {comment} (regarding {obj}).\n"
        else:
            prompt += f"- {subj} {pred} {obj}.\n"
    prompt += "\nBased on the above facts, please provide a detailed, context-aware answer to the question."
    return prompt

#use gpt to generate answer
def generate_answer(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=250
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print("Error calling GPT-4:", e)
        return "An error occurred while generating the answer."


if __name__ == "__main__":

    user_question = input("Please enter your question: ")
    
    retrieved = retrieve_relevant_triples(user_question, G, top_k=5)
    if not retrieved:
        print("No relevant facts found in the knowledge graph.")
    else:
        print("Retrieved relevant triples:")
        for triple in retrieved:
            print(triple)
    
    # construct retrival enhancement
    prompt_text = construct_prompt(user_question, retrieved)
    print("\nConstructed Prompt:\n", prompt_text)
    
    answer_text = generate_answer(prompt_text)
    print("\nGenerated Answer:\n", answer_text)
