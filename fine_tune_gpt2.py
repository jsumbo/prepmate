import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import json
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2FineTuner:
    def __init__(self, model_name="gpt2", dataset_path="./data/waec_qa_dataset.jsonl"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Add padding token to model's vocabulary
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading dataset...")
        data = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Prepare training data
        train_data = []
        for item in data:
            # Format: "Question: {question}\nAnswer: {answer}"
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            train_data.append({"text": text})
        
        # Split into train and validation sets
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def compute_metrics(self, eval_preds):
        """Compute BLEU score and perplexity."""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)
        
        # Calculate BLEU score
        bleu_scores = []
        smoothie = SmoothingFunction().method1
        
        for pred, label in zip(predictions, labels):
            pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
            label_text = self.tokenizer.decode(label, skip_special_tokens=True)
            bleu_scores.append(sentence_bleu([label_text.split()], pred_text.split(), smoothing_function=smoothie))
        
        # Calculate perplexity
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(predictions), torch.tensor(labels))
        perplexity = torch.exp(loss).item()
        
        return {
            "bleu_score": np.mean(bleu_scores),
            "perplexity": perplexity
        }
    
    def train(self, hyperparameters):
        """Train the model with given hyperparameters."""
        logger.info("Starting training...")
        
        # Initialize wandb for experiment tracking
        wandb.init(project="prepmate-gpt2", config=hyperparameters)
        
        # Load and preprocess data
        train_dataset, val_dataset = self.load_and_preprocess_data()
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)
        
        # Initialize model
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        # Resize token embeddings to account for new padding token
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyperparameters["epochs"],
            per_device_train_batch_size=hyperparameters["batch_size"],
            per_device_eval_batch_size=hyperparameters["batch_size"],
            warmup_steps=500,
            weight_decay=hyperparameters["weight_decay"],
            logging_dir="./logs",
            logging_steps=100,
            save_steps=500,
            learning_rate=hyperparameters["learning_rate"],
            report_to="wandb"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        model_save_path = f"./models/gpt2-finetuned-{wandb.run.id}"
        trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        
        # Log final metrics
        final_metrics = trainer.evaluate()
        wandb.log(final_metrics)
        
        return final_metrics

def main():
    # Define hyperparameter search space
    hyperparameters = {
        "learning_rate": 5e-5,
        "batch_size": 4,
        "epochs": 3,
        "weight_decay": 0.01
    }
    
    # Initialize and train
    trainer = GPT2FineTuner()
    metrics = trainer.train(hyperparameters)
    
    logger.info(f"Training completed. Final metrics: {metrics}")

if __name__ == "__main__":
    main()
