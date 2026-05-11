import json
import re

def clean_llm_json(raw_str):
    """Extracts JSON list from potential markdown clutter."""
    try:
        # Look for the first '[' and last ']'
        match = re.search(r'\[.*\]', raw_str, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except:
        return []

def get_gt_entities(sample):
    """
    Extracts (entity, label) pairs from the 'entities' root key 
    found in your specific test.json structure.
    """
    gt_set = set()
    # Your structure uses 'entities' at the root
    for ent in sample.get("entities", []):
        # 'text' is a list in your JSON, e.g., ["Nonylphenol diethoxylate"]
        text = str(ent.get("text", [""])[0]).lower().strip()
        
        # 'semantic_type_id' is a list, e.g., ["T131"]
        labels = ent.get("semantic_type_id", [])
        
        for l in labels:
            if text:
                gt_set.add((text, l))
    return gt_set

def evaluate():
    # 1. LOAD DATA (Ensuring UTF-8 to avoid the previous crash)
    try:
        with open(r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\Segmenting\test.json", "r", encoding="utf-8") as f:
            gt_data = {sample["pmid"]: sample for sample in json.load(f)}

        with open(r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\project\llm_outputs\llama70B_full.json", "r", encoding="utf-8") as f:
            pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    tp, fp, fn = 0, 0, 0

    print(f"{'PMID':<15} | {'GT':<5} | {'Pred':<5} | {'TP':<5}")
    print("-" * 45)

    for pred_item in pred_data:
        pmid = str(pred_item["pmid"])
        if pmid not in gt_data:
            continue

        # Get GT set
        gt_entities = get_gt_entities(gt_data[pmid])
        
        # Get Pred set
        pred_json = clean_llm_json(pred_item["raw_output"])
        pred_entities = set()
        for item in pred_json:
            entity = str(item.get("entity", "")).lower().strip()
            labels = item.get("labels", []) # LLM uses 'labels' per your prompt
            for l in labels:
                if entity:
                    pred_entities.add((entity, l))

        # Intersection math
        current_tp = len(gt_entities.intersection(pred_entities))
        current_fp = len(pred_entities - gt_entities)
        current_fn = len(gt_entities - pred_entities)

        tp += current_tp
        fp += current_fp
        fn += current_fn

        print(f"{pmid:<15} | {len(gt_entities):<5} | {len(pred_entities):<5} | {current_tp:<5}")

    # 2. FINAL CALCULATIONS
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*30)
    print(f"CORRECTED NER RESULTS")
    print("="*30)
    print(f"Total True Positives:  {tp}")
    print(f"Total False Positives: {fp}")
    print(f"Total False Negatives: {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()