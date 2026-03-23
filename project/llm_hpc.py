import pickle
import json

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # loading the LLM and tokenizer
# model_name = "mistralai/Mistral-7B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype=torch.float16
# )



def build_prompt(batch):
    sentences = "\n".join(
        [f"{i+1}. {s}" for i, s in enumerate(batch)]
    )

    return f"""
You are an expert Named Entity Recognition (NER) system.

Extract entities from each sentence using these labels:
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
Example input:
'DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis'

Example output:
Return ONLY valid JSON in this format (example with one entity):
[
  {
    "text": "Pseudomonas aeruginosa",
    "offsets": [[24, 45]],
    "label": "T007"
    
  }
]

Sentences:
{sentences}
"""

#879 abstracts in the test set
with open('./Segmenting/test.pkl', 'rb') as f:
    data = pickle.load(f)

text = []

for sample in data:
    pmid = sample['pmid']
    
    # Extract title and abstract
    title = ""
    abstract = ""
    for passage in sample['passages']:
        if passage['type'] == "title":
            title = passage['text'][0]
        elif passage['type'] == "abstract":
            abstract = passage['text'][0]
    
    # Combine them into one string
    combined_text = title + ". " + abstract  
    
    text.append({
        "pmid": pmid,
        "text": combined_text
    })

# longest text has 3266 characters, which can be expected to have around 800 tokens
longest_text = max(text, key=len)


# iterate through the texts and prompt each of them separately, saving the results in a list
all_responses = []
for t in text:
    prompt = build_prompt([t]['text'])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0
    )

    response = tokenizer.decode(outputs[0])
    
    all_responses.append({
        "id": t['pmid'],
        "response": response
    })

with open("llm_ner_outputs.json", "w", encoding="utf-8") as f:
    json.dump(all_responses, f, ensure_ascii=False, indent=2)