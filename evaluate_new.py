import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

class CustomQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=32):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                question = item['question']
                answer = item['answers'][0]
                self.data.append((question, answer))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        source = self.tokenizer.encode_plus(
            question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        target = self.tokenizer.encode_plus(
            answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze(),
            'answer_text': answer
        }

def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=4,
                early_stopping=True
            )

            # Decode predictions and references
            decoded_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            decoded_refs = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    return predictions, references

def compute_metrics(predictions, references):
    """
    Compute evaluation metrics such as accuracy, BLEU, or ROUGE.
    """
    from datasets import load_metric

    # Example: Using ROUGE for evaluation
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    result = {key: value.mid.fmeasure for key, value in result.items()}

    # Example: Accuracy (exact match)
    exact_matches = sum([1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip()])
    accuracy = exact_matches / len(predictions)

    result["accuracy"] = accuracy
    return result

def main():
    model_name_or_path = "checkpoint/step-3500"  # Path to the last checkpoint
    eval_file_path = "/scratch1/tzhang62/Personalized_RAG/data/convai2/valid.jsonl"  # Evaluation dataset path

    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load evaluation dataset
    eval_dataset = CustomQADataset(file_path=eval_file_path, tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=32)

    # Perform evaluation
    predictions, references = evaluate(model, tokenizer, eval_loader, device)

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Print metrics
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save predictions and references for further analysis
    with open("results/evaluation_results.json", "w") as f:
        json.dump({"predictions": predictions, "references": references, "metrics": metrics}, f, indent=4)

    print("Evaluation complete. Results saved to 'evaluation_results.json'.")

if __name__ == "__main__":
    main()
