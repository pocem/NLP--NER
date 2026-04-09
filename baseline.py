import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

def read_ewt_iob2(file_path):
    sentences = []
    current_tokens = []
    current_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if current_tokens:
                    sentences.append({"tokens": current_tokens, "tags": current_tags})
                    current_tokens, current_tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 3:
                current_tokens.append(parts[1])
                current_tags.append(parts[2])
        if current_tokens:
            sentences.append({"tokens": current_tokens, "tags": current_tags})
                
    return sentences

def tokenize_and_align_labels(examples, tokenizer, tag2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]]) 
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class NER(nn.Module):
    def __init__(self, model_name, num_labels):
        super(NER,self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state
        seq_output = self.dropout(seq_output)
        logits = self.classifier(seq_output)
    
        loss = None
        if labels is not None:
            loss_fun = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fun.ignore_index).type_as(labels)
            )
            loss = loss_fun(active_logits, active_labels)

        return (loss, logits) if loss is not None else logits

def predict(model, dataloader, device, id2tag):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            label_ids = labels.cpu().numpy()
            for i in range(len(preds)):
                sen_pred = [id2tag[p] for p, l in zip(preds[i], label_ids[i]) if l != -100]
                all_predictions.append(sen_pred)
    return all_predictions

def save_formatted_predictions(original_data, predictions, output_file="baseline_results.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(original_data):
            tokens = sentence["tokens"]
            preds = predictions[i]
            for idx, (token, tag) in enumerate(zip(tokens, preds), 1):
                line = f"{idx}\t{token}\t{tag}\t-\t-\n"
                f.write(line)
            f.write("\n")

def main(args):
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    
    print(f"Loading data from {args.train_path} and {args.dev_path}...")
    train_data = read_ewt_iob2(args.train_path)
    dev_data = read_ewt_iob2(args.dev_path)
    
    unique_tags = sorted(list(set(tag for s in train_data for tag in s["tags"])))
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer, "tag2id": tag2id}
    )
    tokenized_dev = dev_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer, "tag2id": tag2id}
    )

    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_dev.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    dev_dataloader = DataLoader(tokenized_dev, batch_size=args.batch_size, collate_fn=data_collator)

    
    print(f"Initializing model: {args.model_name}")
    model = NER(args.model_name, num_labels=len(unique_tags)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model(**batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} complete. Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), args.output_model)
    print(f"Model weights saved to {args.output_model}")

    print("Generating predictions on the dev set...")
    predictions = predict(model, dev_dataloader, device, id2tag)
    save_formatted_predictions(dev_data, predictions, output_file=args.predictions_file)
    print(f"Predictions saved to {args.predictions_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom NER Model with HuggingFace and PyTorch")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train IOB2 file")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to dev IOB2 file")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", help="HuggingFace model repository")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--output_model", type=str, default="ner_model.pth", help="Filename to save the PyTorch model weights")
    parser.add_argument("--predictions_file", type=str, default="baseline_results.txt", help="Filename for the prediction outputs")
    args = parser.parse_args()
    main(args)