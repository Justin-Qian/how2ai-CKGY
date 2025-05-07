import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# Load environment variables
load_dotenv()

class OpenAIBaseline:
    def __init__(self, model_name="gpt-4o"):
        """
        Initialize OpenAI baseline model

        Args:
            model_name (str): Name of the OpenAI model to use
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate_summary(self, text: str, mode: str, annotations: str = None) -> str:
        """
        Generate summary using OpenAI API

        Args:
            text (str): Input text
            mode (str): Summary mode ("B-TEXT", "A-TAG", "A-ONLY", "A-ADD")
            annotations (str, optional): User annotations for A-ADD mode

        Returns:
            str: Generated summary
        """
        if mode == "B-TEXT":
            system_msg = "You are an AI tutor."
            user_msg = f"Please provide a concise and accurate summary of the following academic text:\n\n{text}"
        elif mode == "A-TAG":
            system_msg = "You are an AI tutor helping a student understand annotations."
            user_msg = (
                "Generate a personalized summary that leverages the student's annotations, "
                "explaining highlights (<hl>…</hl>), symbols (<s>…</s>), and comments (<c>…</c>):\n\n"
                f"{text}"
            )
        elif mode == "A-ONLY":
            system_msg = "You are an AI tutor."
            user_msg = (
                "Using only the student's annotations (with <hl>, <s>, <c> tags), "
                "create a personalized summary that explains the key concepts highlighted by those annotations:\n\n"
                f"{text}"
            )
        elif mode == "A-ADD":
            system_msg = "You are an AI tutor helping a student understand the text through their annotations."
            user_msg = (
                "The student has made the following annotations on the text:\n"
                f"{annotations}\n\n"
                "Here is the complete text they were annotating:\n"
                f"{text}\n\n"
                "Please provide a comprehensive summary that incorporates both the student's focus points "
                "from their annotations and the complete context from the text."
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API call error: {str(e)}")
            return ""

def load_data(json_path: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process data from a single document JSON file

    Args:
        json_path (str): Path to JSON file containing a single document

    Returns:
        Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (original_summary, df_b, df_at, df_ao)
    """
    # Load original JSON
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Extract original summary
    original_summary = doc["document"]["summary"]
    doc_id = doc["document"]["id"]
    paragraphs = doc["paragraphs"]
    annotations = doc["annotations"]

    # B-TEXT: Concatenate all paragraphs in reading order
    paras_sorted = sorted(paragraphs, key=lambda p: p["character_start"])
    full_text = "\n".join(p["text"] for p in paras_sorted)

    # A-TAG: Inject tags around each annotated span
    ann_by_para = {}
    for ann in annotations:
        ann_by_para.setdefault(ann["paragraph_id"], []).append(ann)

    tagged_paras = []
    for p in paras_sorted:
        text = p["text"]
        anns = ann_by_para.get(p["id"], [])
        anns_sorted = sorted(anns, key=lambda a: a["referenced_char_start"], reverse=True)

        for a in anns_sorted:
            start = a["referenced_char_start"] - p["character_start"]
            end = a["referenced_char_end"] - p["character_start"]

            if a["type"] == "highlight":
                ot, ct = "<hl>", "</hl>"
            elif a["type"] == "comment":
                ot, ct = "<c>", "</c>"
            elif a["type"] == "symbol":
                ot, ct = "<s>", "</s>"
            else:
                continue

            text = text[:start] + ot + text[start:end] + ct + text[end:]

        tagged_paras.append(text)

    tagged_text = "\n".join(tagged_paras)

    # A-ONLY: Get user-supplied annotated_text in order of appearance
    anns_sorted_all = sorted(annotations, key=lambda a: a["referenced_char_start"])
    only_texts = [a["annotated_text"] for a in anns_sorted_all]
    a_only_text = " ".join(only_texts)

    # Create single-row DataFrames
    df_b = pd.DataFrame([{"doc_id": doc_id, "text": full_text}])
    df_at = pd.DataFrame([{"doc_id": doc_id, "text": tagged_text}])
    df_ao = pd.DataFrame([{"doc_id": doc_id, "annotations": a_only_text}])

    return original_summary, df_b, df_at, df_ao

def generate_all_summaries(df_b: pd.DataFrame, df_at: pd.DataFrame,
                         df_ao: pd.DataFrame, baseline: OpenAIBaseline) -> Dict[str, pd.DataFrame]:
    """
    Generate summaries using different modes

    Args:
        df_b (pd.DataFrame): B-TEXT dataset
        df_at (pd.DataFrame): A-TAG dataset
        df_ao (pd.DataFrame): A-ONLY dataset
        baseline (OpenAIBaseline): OpenAI baseline model instance

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each mode
    """
    doc_id = df_b["doc_id"].iloc[0]
    txt_b = df_b["text"].iloc[0]
    txt_at = df_at["text"].iloc[0]
    ann_o = df_ao["annotations"].iloc[0]

    # Generate summaries
    s_b = baseline.generate_summary(txt_b, "B-TEXT")
    s_at = baseline.generate_summary(txt_at, "A-TAG")
    s_ao = baseline.generate_summary(ann_o, "A-ONLY")
    s_add = baseline.generate_summary(txt_b, "A-ADD", annotations=ann_o)

    # Create DataFrames for each mode
    dfs = {
        "B-TEXT": pd.DataFrame([{
            "doc_id": doc_id,
            "mode": "B-TEXT",
            "generated_summary": s_b
        }]),
        "A-TAG": pd.DataFrame([{
            "doc_id": doc_id,
            "mode": "A-TAG",
            "generated_summary": s_at
        }]),
        "A-ONLY": pd.DataFrame([{
            "doc_id": doc_id,
            "mode": "A-ONLY",
            "generated_summary": s_ao
        }]),
        "A-ADD": pd.DataFrame([{
            "doc_id": doc_id,
            "mode": "A-ADD",
            "generated_summary": s_add
        }])
    }

    return dfs

def main():
    # Set paths
    data_dir = "data"
    output_dir = "generated_summary"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize baseline model
    baseline = OpenAIBaseline()

    # Get all JSON files in the data directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    print(f"Found {len(json_files)} JSON files to process")

    # Initialize DataFrames for each mode
    mode_dfs = {
        "ORIGINAL": pd.DataFrame(columns=["doc_id", "mode", "generated_summary"]),
        "B-TEXT": pd.DataFrame(columns=["doc_id", "mode", "generated_summary"]),
        "A-TAG": pd.DataFrame(columns=["doc_id", "mode", "generated_summary"]),
        "A-ONLY": pd.DataFrame(columns=["doc_id", "mode", "generated_summary"]),
        "A-ADD": pd.DataFrame(columns=["doc_id", "mode", "generated_summary"])
    }

    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        json_filename = os.path.splitext(json_file)[0]

        print(f"\nProcessing {json_file}...")

        try:
            # Load data
            print("Loading data...")
            original_summary, df_b, df_at, df_ao = load_data(json_path)

            # Add original summary
            mode_dfs["ORIGINAL"] = pd.concat([
                mode_dfs["ORIGINAL"],
                pd.DataFrame([{
                    "doc_id": df_b["doc_id"].iloc[0],
                    "mode": "ORIGINAL",
                    "generated_summary": original_summary
                }])
            ], ignore_index=True)

            # Generate summaries
            print("Generating summaries...")
            summaries = generate_all_summaries(df_b, df_at, df_ao, baseline)

            # Add generated summaries to respective DataFrames
            for mode, df in summaries.items():
                mode_dfs[mode] = pd.concat([mode_dfs[mode], df], ignore_index=True)

            print(f"✅ Completed {json_file}")

        except Exception as e:
            print(f"❌ Error processing {json_file}: {str(e)}")
            continue

    # Save results for each mode
    print("\nSaving results...")
    for mode, df in mode_dfs.items():
        if not df.empty:  # Only save if we have records for this mode
            output_file = os.path.join(output_dir, f"{mode.replace('-', '_')}.csv")
            df.to_csv(output_file, index=False)
            print(f"✅ Saved {mode} results to {output_file}")

    print("\nAll files processed!")

if __name__ == "__main__":
    main()
