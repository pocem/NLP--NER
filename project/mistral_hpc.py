import pickle
import json
from xml.parsers.expat import model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from huggingface_hub import login


login(token=os.environ["HF_TOKEN"])

def build_prompt(batch):
    sentences = "\n".join([f"Sentence {i+1}: {s}" for i, s in enumerate(batch)])
    
    return f"""Extract biomedical entities from sentences. Return JSON array: [{{"entity":str,"labels":list,"offsets":[start,end]}}]

Valid label codes:
T001,Organism T002,Plant T004,Fungus T005,Virus T007,Bacterium T008,Animal T010,Vertebrate T011,Amphibian T012,Bird T013,Fish T014,Reptile T015,Mammal T016,Human
T017,Anatomical Structure T018,Embryonic Structure T019,Congenital Abnormality T020,Acquired Abnormality T021,Fully Formed Anatomical Structure T022,Body System T023,Body Part T024,Tissue T025,Cell T026,Cell Component
T028,Gene or Genome T029,Body Location T030,Body Space T031,Body Substance T032,Organism Attribute
T033,Finding T034,Lab Result T037,Injury T038,Biologic Function T039,Physiologic Function T040,Organism Function T041,Mental Process T042,Organ Function T043,Cell Function T044,Molecular Function T045,Genetic Function T046,Pathologic Function T047,Disease T048,Mental Dysfunction T049,Cell Dysfunction T050,Experimental Disease Model
T051,Event T052,Activity T053,Behavior T054,Social Behavior T055,Individual Behavior T056,Daily Activity T057,Occupational Activity T058,Health Care Activity T059,Lab Procedure T060,Diagnostic Procedure T061,Therapeutic Procedure T062,Research Activity T063,Molecular Biology Technique
T064,Regulatory Activity T065,Educational Activity T066,Machine Activity T067,Phenomenon T068,Human-caused Phenomenon T069,Environmental Effect T070,Natural Phenomenon
T071,Entity T072,Physical Object T073,Manufactured Object T074,Medical Device T075,Research Device
T077,Conceptual Entity T078,Idea T079,Temporal Concept T080,Qualitative Concept T081,Quantitative Concept T082,Spatial Concept T083,Geographic Area
T085,Molecular Sequence T086,Nucleotide Sequence T087,Amino Acid Sequence T088,Carbohydrate Sequence
T089,Regulation T090,Occupation T091,Biomedical Occupation T092,Organization T093,Healthcare Organization T094,Professional Society T095,Relief Organization T096,Group T097,Professional Group T098,Population Group T099,Family Group T100,Age Group T101,Patient Group T102,Group Attribute
T103,Chemical T104,Chemical Structure T109,Organic Chemical T114,Nucleic Acid T116,Amino Acid T120,Chemical Function T121,Pharmacologic Substance T122,Biomedical Material T123,Bioactive Substance T125,Hormone T126,Enzyme T127,Vitamin T129,Immunologic Factor T130,Diagnostic Aid T131,Hazardous Substance
T167,Substance T168,Food T169,Functional Concept T170,Intellectual Product T171,Language
T184,Sign or Symptom T185,Classification T190,Anatomical Abnormality T191,Neoplastic Process T192,Receptor T194,Archaeon T195,Antibiotic T196,Element/Ion/Isotope T197,Inorganic Chemical
T200,Clinical Drug T201,Clinical Attribute T203,Drug Delivery Device T204,Eukaryote

RULES:
- no explanations
- no markdown
- no empty lists
- be strict JSON only

EXAMPLE:
text: "bacterial infection in mice" → [{{"entity":"bacterial infection","labels":["T007","T047"],"offsets":[0,19]}},{{"entity":"mice","labels":["T008"],"offsets":[23,27]}}]

Sentences:
{sentences}"""


def main():

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()

    print("Model loaded")
    print("CUDA:", torch.cuda.is_available())

    # -------- LOAD DATA --------
    with open("./Segmenting/test.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:3]

    # -------- PREPROCESS --------
    text = []
    for sample in data:
        pmid = sample["pmid"]

        title = ""
        abstract = ""

        for passage in sample["passages"]:
            if passage["type"] == "title":
                title = passage["text"][0]
            elif passage["type"] == "abstract":
                abstract = passage["text"][0]

        combined = title + ". " + abstract

        text.append({
            "pmid": pmid,
            "text": combined
        })

    # -------- LOOP --------
    all_outputs = []

    for item in text:

        pmid = item["pmid"]
        prompt = build_prompt([item["text"]])

        print(f"\nProcessing PMID: {pmid}")

        # 🔥 SIMPLE TOKENIZATION (NO CHAT TEMPLATE)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        print("\nRAW OUTPUT:\n")
        print(response[:500])
        print("\n----------------------\n")

        all_outputs.append({
            "pmid": pmid,
            "raw_output": response
        })

    # -------- SAVE --------
    out_dir = "/home/migre/NER_PROJECT/llm_outputs"
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "mistral_ner_outputs.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

    print("Saved to:", output_path)


if __name__ == "__main__":
    main()