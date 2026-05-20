import json
import re
import os
from collections import Counter

def clean_llm_json(raw_str):
    """Extracts JSON list from potential markdown clutter."""
    try:
        match = re.search(r'\[.*\]', raw_str, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except:
        return []

def get_gt_entities_with_tags(sample):
    """Returns a dict mapping {entity_text: set_of_tags} for the ground truth."""
    gt_map = {}
    for ent in sample.get("entities", []):
        text_list = ent.get("text", [""])
        text = str(text_list[0]).lower().strip() if text_list else ""
        labels = ent.get("semantic_type_id", [])
        
        if text:
            if text not in gt_map:
                gt_map[text] = set()
            for l in labels:
                gt_map[text].add(l)
    return gt_map

def analyze_qualitative_by_percentage(min_gt_threshold=50):
    gt_path = r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\Segmenting\test.json"
    pred_path = r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\project\llm_outputs\llama70B_fewshot.json"
    
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = {sample["pmid"]: sample for sample in json.load(f)}
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    missed_tags_counter = Counter()
    total_gt_tags_counter = Counter()

    # Process abstract by abstract
    for pred_item in pred_data:
        pmid = str(pred_item["pmid"])
        if pmid not in gt_data:
            continue
            
        gt_map = get_gt_entities_with_tags(gt_data[pmid])
        
        pred_json = clean_llm_json(pred_item["raw_output"])
        pred_texts = {str(item.get("entity", "")).lower().strip() for item in pred_json if item.get("entity")}
        
        # Count total occurrences of tags in the ground truth
        for text, tags in gt_map.items():
            for t in tags:
                total_gt_tags_counter[t] += 1

        # Find which ground truth entities were missed
        for gt_text, gt_tags in gt_map.items():
            if gt_text not in pred_texts:
                for tag in gt_tags:
                    missed_tags_counter[tag] += 1

    # Calculate miss percentages for tags appearing at least 50 times
    tag_stats = []
    
    for tag, total_count in total_gt_tags_counter.items():
        if total_count >= min_gt_threshold:
            missed_count = missed_tags_counter[tag]
            miss_percentage = (missed_count / total_count) * 100
            tag_stats.append({
                "tag": tag,
                "miss_percentage": miss_percentage,
                "missed_count": missed_count,
                "total_count": total_count
            })
    
    # Sort by miss percentage descending
    tag_stats.sort(key=lambda x: x["miss_percentage"], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print(f"TAGS WITH ≥ {min_gt_threshold} MENTIONS - SORTED BY MISS PERCENTAGE")
    print("="*80)
    print(f"{'Rank':<6} {'Tag':<15} {'Miss %':<12} {'Missed/Total':<15}")
    print("-"*80)
    
    for idx, stat in enumerate(tag_stats, 1):
        print(f"{idx:<6} {stat['tag']:<15} {stat['miss_percentage']:.2f}%      {stat['missed_count']}/{stat['total_count']}")
    
    print("="*80)
    
    # Summary stats
    if tag_stats:
        avg_miss = sum(s["miss_percentage"] for s in tag_stats) / len(tag_stats)
        print(f"\nSummary:")
        print(f"Total tags analyzed: {len(tag_stats)}")
        print(f"Average miss percentage: {avg_miss:.2f}%")

if __name__ == '__main__':
    analyze_qualitative_by_percentage(min_gt_threshold=50)