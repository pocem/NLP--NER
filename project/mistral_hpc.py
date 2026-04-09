import pickle
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def build_prompt(batch):
    sentences = "\n".join([f"{i+1}. {s}" for i, s in enumerate(batch)])
    return f"""
You are an expert Named Entity Recognition (NER) system.

Extract entities from each sentence using these labels from PubMed:
T001,Organism
T002,Plant
T004,Fungus
T005,Virus
T007,Bacterium
T008,Animal
T010,Vertebrate
T011,Amphibian
T012,Bird
T013,Fish
T014,Reptile
T015,Mammal
T016,Human
T017,Anatomical Structure
T018,Embryonic Structure
T019,Congenital Abnormality
T020,Acquired Abnormality
T021,Fully Formed Anatomical Structure
T022,Body System
T023,"Body Part, Organ, or Organ Component"
T024,Tissue
T025,Cell
T026,Cell Component
T028,Gene or Genome
T029,Body Location or Region
T030,Body Space or Junction
T031,Body Substance
T032,Organism Attribute
T033,Finding
T034,Laboratory or Test Result
T037,Injury or Poisoning
T038,Biologic Function
T039,Physiologic Function
T040,Organism Function
T041,Mental Process
T042,Organ or Tissue Function
T043,Cell Function
T044,Molecular Function
T045,Genetic Function
T046,Pathologic Function
T047,Disease or Syndrome
T048,Mental or Behavioral Dysfunction
T049,Cell or Molecular Dysfunction
T050,Experimental Model of Disease
T051,Event
T052,Activity
T053,Behavior
T054,Social Behavior
T055,Individual Behavior
T056,Daily or Recreational Activity
T057,Occupational Activity
T058,Health Care Activity
T059,Laboratory Procedure
T060,Diagnostic Procedure
T061,Therapeutic or Preventive Procedure
T062,Research Activity
T063,Molecular Biology Research Technique
T064,Governmental or Regulatory Activity
T065,Educational Activity
T066,Machine Activity
T067,Phenomenon or Process
T068,Human-caused Phenomenon or Process
T069,Environmental Effect of Humans
T070,Natural Phenomenon or Process
T071,Entity
T072,Physical Object
T073,Manufactured Object
T074,Medical Device
T075,Research Device
T077,Conceptual Entity
T078,Idea or Concept
T079,Temporal Concept
T080,Qualitative Concept
T081,Quantitative Concept
T082,Spatial Concept
T083,Geographic Area
T085,Molecular Sequence
T086,Nucleotide Sequence
T087,Amino Acid Sequence
T088,Carbohydrate Sequence
T089,Regulation or Law
T090,Occupation or Discipline
T091,Biomedical Occupation or Discipline
T092,Organization
T093,Health Care Related Organization
T094,Professional Society
T095,Self-help or Relief Organization
T096,Group
T097,Professional or Occupational Group
T098,Population Group
T099,Family Group
T100,Age Group
T101,Patient or Disabled Group
T102,Group Attribute
T103,Chemical
T104,Chemical Viewed Structurally
T109,Organic Chemical
T114,"Nucleic Acid, Nucleoside, or Nucleotide"
T116,"Amino Acid, Peptide, or Protein"
T120,Chemical Viewed Functionally
T121,Pharmacologic Substance
T122,Biomedical or Dental Material
T123,Biologically Active Substance
T125,Hormone
T126,Enzyme
T127,Vitamin
T129,Immunologic Factor
T130,"Indicator, Reagent, or Diagnostic Aid"
T131,Hazardous or Poisonous Substance
T167,Substance
T168,Food
T169,Functional Concept
T170,Intellectual Product
T171,Language
T184,Sign or Symptom
T185,Classification
T190,Anatomical Abnormality
T191,Neoplastic Process
T192,Receptor
T194,Archaeon
T195,Antibiotic
T196,"Element, Ion, or Isotope"
T197,Inorganic Chemical
T200,Clinical Drug
T201,Clinical Attribute
T203,Drug Delivery Device
T204,Eukaryote

Return ONLY valid JSON.
Return a JSON array of entities found in the text.

Each entity must have:
- "entity": string
- "labels": list of label codes
- "offsets": list of [start, end]

If no entities exist, return an empty list.

Sentences:
{sentences}
"""

def safe_parse_json(text):
    """Attempt to parse JSON returned by LLM; return empty list if it fails."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # clean up common issues
        cleaned = text.replace('\n', '').replace('\t', '').replace(',]', ']').replace(',}', '}')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("Failed to parse JSON, returning empty list")
            return []

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
        print("Response snippet:", response[:100])

        # Parse JSON safely
        parsed_entities = safe_parse_json(response)

        for pmid in batch_pmids:
            all_responses.append({
                "pmid": pmid,
                "entities": parsed_entities
            })

    # Save to JSON
    with open("llm_outputs/mistral_ner_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)
    print("All responses saved to mistral_ner_outputs.json")

if __name__ == "__main__":
    main()