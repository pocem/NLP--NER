import pickle
import json
import requests

print("Script started")

# --- LLM call ---
def query_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        }
    )
    return response.json()["response"]


# --- build prompt ONLY ---
def build_prompt(batch):
    sentences = "\n".join(
        [f"{i+1}. {s}" for i, s in enumerate(batch)]
    )

    return f"""
You are an expert Named Entity Recognition (NER) system.


Extract entities from each sentence using these labels:
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

Sentences:
{sentences}
"""


# --- load data ---
with open('C:\\Users\\misog\\SCHOOL\\4th semester\\Natural language processing\\project\\Segmenting\\test.pkl', 'rb') as f:
    data = pickle.load(f)

print("Data loaded:", len(data))
# --- build text list ---
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

    text.append({
        "pmid": pmid,
        "text": combined_text
    })


# --- main loop (SMALL TEST) ---
all_responses = []

for t in text[:2]:  # keep it small
    print("Processing:", t["pmid"])

    prompt = build_prompt([t["text"]])
    
    response = query_llm(prompt)
    
    print("Response:", response[:200])  # print first 200 chars

    all_responses.append({
        "pmid": t["pmid"],
        "response": response
    })


# --- save ---
with open("llm_outputs/tinyllama_ner_outputs.json", "w", encoding="utf-8") as f:
    json.dump(all_responses, f, ensure_ascii=False, indent=2)