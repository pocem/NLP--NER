import pickle
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def build_prompt(batch):
    sentences = "\n".join([f"{i+1}. {s}" for i, s in enumerate(batch)])
    return f"""
You are an expert Named Entity Recognition (NER) system.

Extract entities from each sentence using these labels from PubMed:
T005: Virus (level 4)
T007: Bacterium (level 4)
T017: Anatomical Structure (level 3)
T022: Body System (level 5)
T031: Body Substance (level 4)
T033: Finding (level 3)
T037: Injury or Poisoning (level 3)
T038: Biologic Function (level 4)
T058: Health Care Activity (level 4)
T062: Research Activity (level 4)
T074: Medical Device (level 4)
T082: Spatial Concept (level 4)
T091: Biomedical Occupation or Discipline (level 4)
T092: Organization (level 3)
T097: Professional or Occupational Group (level 4)
T098: Population Group (level 4)
T103: Chemical (level 4)
T168: Food (level 4)
T170: Intellectual Product (level 3)
T201: Clinical Attribute (level 4)
T204: Eukaryote (level 4)

If no entities exist, return an empty list.
There can be more than one label per entity.
Return ONLY valid JSON.

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
    batch_size = 5
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
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response snippet:", response[:200])

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