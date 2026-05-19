import json
import re
import os

def clean_llm_json(raw_str):
    """Extracts JSON list from potential markdown clutter."""
    try:
        match = re.search(r'\[.*\]', raw_str, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except:
        return []

def get_gt_entities(sample):
    """Extracts (entity, label) pairs from the 'entities' root key."""
    gt_set = set()
    for ent in sample.get("entities", []):
        text = str(ent.get("text", [""])[0]).lower().strip()
        labels = ent.get("semantic_type_id", [])
        for l in labels:
            if text:
                gt_set.add((text, l))
    return gt_set

def evaluate():
    # File Paths
    gt_path = r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\Segmenting\test.json"
    pred_path = r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\project\llm_outputs\llama70B_fewshot.json"
    output_txt_path = "llm_result_fewshot.txt"

    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = {sample["pmid"]: sample for sample in json.load(f)}

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    tp, fp, fn = 0, 0, 0
    
    # Open text file for writing
    with open(output_txt_path, "w", encoding="utf-8") as out_f:
        header = f"{'PMID':<15} | {'GT':<5} | {'Pred':<5} | {'TP':<5}\n"
        separator = "-" * 45 + "\n"
        out_f.write(header)
        out_f.write(separator)
        
        # Also print to console so you can see it happening
        print(header, end="")
        print(separator, end="")

        for pred_item in pred_data:
            pmid = str(pred_item["pmid"])
            if pmid not in gt_data:
                continue

            gt_entities = get_gt_entities(gt_data[pmid])
            pred_json = clean_llm_json(pred_item["raw_output"])
            pred_entities = set()
            for item in pred_json:
                entity = str(item.get("entity", "")).lower().strip()
                labels = item.get("labels", [])
                for l in labels:
                    if entity:
                        pred_entities.add((entity, l))

            current_tp = len(gt_entities.intersection(pred_entities))
            current_fp = len(pred_entities - gt_entities)
            current_fn = len(gt_entities - pred_entities)

            tp += current_tp
            fp += current_fp
            fn += current_fn

            row = f"{pmid:<15} | {len(gt_entities):<5} | {len(pred_entities):<5} | {current_tp:<5}\n"
            out_f.write(row)
            print(row, end="")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        summary = (
            "\n" + "="*30 + "\n"
            "CORRECTED NER RESULTS\n"
            + "="*30 + "\n"
            f"Total True Positives:  {tp}\n"
            f"Total False Positives: {fp}\n"
            f"Total False Negatives: {fn}\n"
            + "-" * 30 + "\n"
            f"Precision: {precision:.4f}\n"
            f"Recall:    {recall:.4f}\n"
            f"F1 Score:  {f1:.4f}\n"
            + "="*30 + "\n"
        )
        out_f.write(summary)
        print(summary)
    
    print(f"\n[DONE] Results successfully saved to: {os.path.abspath(output_txt_path)}")

if __name__ == '__main__':
    evaluate()