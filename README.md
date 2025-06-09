# PrepMate: WASSCE Exam Preparation Assistant

PrepMate is an AI-powered chatbot designed to help students prepare for the West African Senior School Certificate Examination (WASSCE). It uses a fine-tuned GPT-2 model to provide accurate and helpful answers to exam-related questions.

## Features

- Domain-specific question answering for WASSCE subjects
- Similarity matching with existing exam questions
- Confidence scoring for answers
- User feedback mechanism
- Performance tracking with BLEU scores and perplexity metrics

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
```

4. Set up Weights & Biases (for experiment tracking):
```bash
wandb login
```

## Usage

### Training the Model

To fine-tune the GPT-2 model on your dataset:

```bash
python fine_tune_gpt2.py
```

This will:
- Load and preprocess the dataset
- Fine-tune the GPT-2 model
- Track metrics using Weights & Biases
- Save the best model to the `models` directory

### Running the Chatbot

To start the Streamlit interface:

```bash
streamlit run app.py
```

## Dataset Structure

The dataset should be in JSONL format with the following structure:
```json
{
    "question": "Your question here",
    "answer": "The answer to the question",
    "subject": "Subject name (optional)",
    "explanation": "Detailed explanation (optional)"
}
```

## Model Performance

The model is evaluated using:
- BLEU score for answer quality
- Perplexity for model confidence
- Similarity matching with existing questions

## Project Structure

```
prepmate/
├── app.py              # Streamlit web interface
├── fine_tune_gpt2.py   # Model training script
├── requirements.txt    # Project dependencies
├── data/              # Dataset directory
│   └── waec_qa_dataset.jsonl
├── models/            # Saved model checkpoints
└── results/           # Training results and logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

