import json
import os
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Paths
DATA_PATH = "./data/waec_qa_dataset.jsonl"
MODEL_OUTPUT_DIR = "./models/prepmate_gpt2"

# Load dataset from JSONL file
def load_qa_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Format input as Q + A + explanation
            text = f"Subject: {item['subject']}\nQuestion: {item['question']}\nAnswer: {item['answer']}\nExplanation: {item['explanation']}\n"
            data.append({"text": text})
    return Dataset.from_list(data)

def main():
    # Load dataset
    dataset = load_qa_dataset(DATA_PATH)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token, so set eos as pad

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    main()
