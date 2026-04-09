import pickle
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def build_prompt(batch):
    sentences = "\n".join([f"{i+1}. {s}" for i, s in enumerate(batch)])
    return f"""
You are an expert Named Entity Recognition (NER) system.

Extract entities from each sentence using these labels from PubMed:
T005: Virus
T007: Bacterium
T017: Anatomical Structure
T022: Body System
T031: Body Substance
T033: Finding
T037: Injury or Poisoning
T038: Biologic Function
T058: Health Care Activity
T062: Research Activity
T074: Medical Device
T082: Spatial Concept
T091: Biomedical Occupation or Discipline
T092: Organization
T097: Professional or Occupational Group
T098: Population Group
T103: Chemical
T168: Food
T170: Intellectual Product
T201: Clinical Attribute
T204: Eukaryote

Return ONLY valid JSON.
Return a JSON array of entities found in the text.

Each entity must have:
- "entity": string
- "labels": list of label codes
- "offsets": list of [start, end]

If no entities exist, return an empty list.

Example input sentences:
"Some text about COVID-19."
Example output format:
{
    "entity": "COVID-19",
    "labels": ["T005", "T033"]
    "offsets": [[16, 23]]
}

Sentences:
{sentences}
"""

def main():
    # Load tokenizer and model
    model_name = "mistralai/Mistral-7B-Instruct"
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    print("Model loaded!")

    # Load test data
    with open('./Segmenting/test.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded: {len(data)} samples")

    # Prepare text
    text = []
    for sample in data:
        pmid = sample['pmid']
        title = ""
        abstract = ""
        for passage in sample['passages']:
            if passage['type'] == "title":
                title = passage['text'][0]
            elif passage['type'] == "abstract":
                abstract = passage['text'][0]
        combined_text = title + ". " + abstract
        text.append({"pmid": pmid, "text": combined_text})

    # --- batching ---
    batch_size = 1
    all_responses = []

    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        batch_texts = [t["text"] for t in batch]
        batch_pmids = [t["pmid"] for t in batch]

        print(f"Processing batch {i//batch_size + 1}: PMIDs {batch_pmids}")
        prompt = build_prompt(batch_texts)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0
        )
        response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
        )
        print("Response snippet:", response[:50])

        # store same response for each PMID (you can split later if needed)
        for pmid in batch_pmids:
            all_responses.append({
                "pmid": pmid,
                "response": response
            })

    # Save to JSON
    with open("llm_outputs/mistral_ner_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)
    print("All responses saved to mistral_ner_outputs.json")

if __name__ == "__main__":
    main()