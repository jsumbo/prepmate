# PrepMate: WASSCE Exam Preparation Assistant

PrepMate is an AI-powered chatbot designed to help students prepare for the West African Senior School Certificate Examination (WASSCE). It leverages a domain-specific dataset and a transformer-based language model to provide accurate, helpful answers to exam-related questions in Math, English, Science, and more.

## 🚀 Live Demo

Try the app here: [https://prepmate.streamlit.app/](https://prepmate.streamlit.app/)

---

## Features
- Domain-specific question answering for WASSCE subjects
- Similarity matching with existing exam questions
- Confidence scoring for answers (progress bar and badges)
- User feedback mechanism (thumbs up/down, suggestions)
- Example/sample questions for user guidance
- Warning and fallback for out-of-domain queries
- Clean, modern Streamlit interface with icons and loading spinners

## Project Structure
```
prepmate/
├── app.py                # Streamlit web interface
├── fine_tune_gpt2.py     # Model training script (if using local model)
├── requirements.txt      # Project dependencies
├── data/                 # Dataset directory
│   └── waec_qa_dataset.jsonl
├── models/               # Saved model checkpoints (gitignored)
├── README.md             # Project documentation
└── ...
```

## Dataset
- Format: JSONL, each line is a JSON object with at least a `question` and `answer` field.
- Example:
```json
{"question": "What is the chemical formula for water?", "answer": "H2O"}
```
- Used for similarity matching, sample questions, and (optionally) model fine-tuning.

## How It Works
1. **User submits a question**
2. **Similarity check**: The app compares the question to those in the dataset
3. **If similar**: The model generates an answer and displays confidence
4. **If not similar**: The app warns the user and shows sample/example questions
5. **Feedback**: Users can rate answers and provide suggestions

## Running Locally
1. Clone the repo and `cd prepmate`
2. Create a virtual environment and activate it
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Model
- Uses a transformer-based model (e.g., GPT-2) via Hugging Face Transformers
- Can be extended to use OpenAI API for more powerful responses
- Model is guided by the domain-specific dataset for relevance

## Evaluation
- BLEU score and perplexity (if fine-tuning locally)
- Qualitative: User feedback, confidence scores, and similarity checks



