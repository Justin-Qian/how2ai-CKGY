from typing import List, Dict, Tuple

def extract_triplets(annotation_json: Dict) -> List[Tuple[str, str, str]]:
    print(triplets)
    triplets = []

    for user, docs in annotation_json.items():
        user_node = f"User:{user}"

        for doc_uri, doc_data in docs.items():
            doc_node = f"Document:{doc_uri}"
            doc_title = doc_data.get("title", "Untitled Document")

            triplets.append((doc_node, "has_title", doc_title))

            for ann in doc_data.get("annotations", []):
                ann_id = ann["id"]
                ann_node = f"Annotation:{ann_id}"

                triplets.extend([
                    (user_node, "created_annotation", ann_node),
                    (ann_node, "highlights_text", ann.get("highlighted_text", "")),
                    (ann_node, "has_comment", ann.get("comment", "")),
                    (ann_node, "created_on", ann["created"]),
                    (ann_node, "appears_in_document", doc_node),
                ])

                pos = ann.get("position", {})
                context = ann.get("context", {})
                triplets.extend([
                    (ann_node, "has_position", f"{pos.get('start_char')}-{pos.get('end_char')}"),
                    (ann_node, "has_preceding_text", context.get("preceding_text", "")),
                    (ann_node, "has_following_text", context.get("following_text", "")),
                ])

                if "annotation_type" in ann:
                    triplets.append((ann_node, "annotation_type", ann["annotation_type"]))

                if "target_type" in ann:
                    triplets.append((ann_node, "target_type", ann["target_type"]))

                if "tags" in ann:
                    for tag in ann["tags"]:
                        triplets.append((ann_node, "tagged_with", tag))

    return triplets
