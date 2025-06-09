# PrepMate: WASSCE Exam Preparation Assistant

## Overview
PrepMate is a domain-specific chatbot that helps Liberian students prepare for WASSCE exams by answering questions and providing explanations across key subjects.

## Directory Structure
- `data/waec_qa_dataset.jsonl`: Dataset of Q&A pairs with explanations
- `models/`: Folder to store the fine-tuned GPT-2 model
- `fine_tune_gpt2.py`: Script to fine-tune GPT-2 on the dataset
- `app.py`: Streamlit user interface for interacting with the chatbot
- `requirements.txt`: Python dependencies

## Setup Instructions

1. **Clone the repository or copy files**

2. **Create and activate a Python environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
