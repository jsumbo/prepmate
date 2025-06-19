import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json
import numpy as np
from collections import Counter
import math
import openai
from dotenv import load_dotenv
import os

MODEL_DIR = "gpt2"  # Using the base GPT-2 model from Hugging Face
DATASET_PATH = "./data/waec_qa_dataset.jsonl"

SIMILARITY_THRESHOLD = 0.9  # strict threshold for in-domain

EXAMPLE_QUESTIONS = [
    "What is the chemical formula for water?",
    "Solve for x: 2x + 3 = 7.",
    "Who wrote the play 'Romeo and Juliet'?",
    "Explain the process of photosynthesis.",
    "What is the capital of Ghana?"
]

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_dataset():
    questions = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
    return questions

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    # Add padding token to model's vocabulary
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    # Resize token embeddings to account for new padding token
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return tokenizer, model

def get_word_frequencies(text):
    words = text.lower().split()
    return Counter(words)

def calculate_similarity(text1, text2):
    # Get word frequencies
    freq1 = get_word_frequencies(text1)
    freq2 = get_word_frequencies(text2)
    
    # Get all unique words
    all_words = set(freq1.keys()) | set(freq2.keys())
    
    # Calculate dot product and magnitudes
    dot_product = sum(freq1[word] * freq2[word] for word in all_words)
    magnitude1 = math.sqrt(sum(freq1[word] ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(freq2[word] ** 2 for word in all_words))
    
    # Calculate cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def find_similar_questions(user_question, dataset_questions, threshold=0.3):
    similarities = [(q, calculate_similarity(user_question, q)) for q in dataset_questions]
    similar_questions = [(q, s) for q, s in similarities if s > threshold]
    return sorted(similar_questions, key=lambda x: x[1], reverse=True)

def generate_response(tokenizer, model, prompt, max_length=150):
    try:
        # Encode the input with attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate response with proper attention mask
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Calculate confidence score based on token probabilities
        token_probs = torch.softmax(outputs.scores[0], dim=-1)
        confidence = float(torch.mean(token_probs.max(dim=-1)[0]))
        
        text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = text[len(prompt):].strip()
        
        return response, confidence
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response. Please try again.", 0.0

def get_chatgpt_response(prompt, model="gpt-3.5-turbo"):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ChatGPT Error] {str(e)}"

def main():
    st.set_page_config(page_title="PrepMate: WASSCE Exam Preparation Assistant", page_icon="📚")
    st.title("📚 PrepMate: WASSCE Exam Preparation Assistant")
    st.write("Ace your WASSCE prep — ask questions in Math, English, Science, or more, and get answers with easy-to-understand explanations.")

    # Load model and dataset
    tokenizer, model = load_model()
    dataset_questions = load_dataset()

    user_input = st.text_input("Your question:")

    if st.button("Get Answer"):
        if user_input.strip():
            # Calculate similarity with dataset
            similarities = [calculate_similarity(user_input, q) for q in dataset_questions]
            max_similarity = max(similarities) if similarities else 0
            similar_questions = find_similar_questions(user_input, dataset_questions)

            # Strict check: only answer if a similar question is found
            if max_similarity >= SIMILARITY_THRESHOLD:
                # Find the most similar question
                best_idx = similarities.index(max_similarity)
                matched_question = dataset_questions[best_idx]
                # Generate response
                prompt = f"Question: {matched_question}\nAnswer:"
                with st.spinner("PrepMate is thinking..."):
                    answer, confidence = generate_response(tokenizer, model, prompt)
                st.markdown(f"**Answer:** {answer}")
                # Show confidence level with progress bar and badge
                st.markdown(f"**Confidence Score:**")
                st.progress(confidence, text=f"{confidence*100:.1f}%")
                if confidence >= 0.8:
                    st.success("High confidence")
                elif confidence >= 0.6:
                    st.info("Moderate confidence")
                else:
                    st.warning("Low confidence")
                # Show similar questions if available
                if similar_questions:
                    st.markdown("**Similar questions in our database:**")
                    for question, score in similar_questions[:3]:
                        st.markdown(f"- {question} (Similarity: {score:.2f})")
                # Add feedback mechanism
                st.markdown("---")
                st.markdown("Was this answer helpful?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Yes"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 No"):
                        st.error("We're sorry the answer wasn't helpful. We'll use your feedback to improve.")
            else:
                # If not in dataset, use ChatGPT
                with st.spinner("Prepmate is thinking..."):
                    chatgpt_answer = get_chatgpt_response(user_input)
                st.markdown(f"**ChatGPT Answer:** {chatgpt_answer}")
                st.info("This answer was generated by ChatGPT because your question was not found in our training data.")
                if similar_questions:
                    st.markdown("**Sample questions in our database:**")
                    for question, score in similar_questions[:3]:
                        st.markdown(f"- {question} (Similarity: {score:.2f})")
                else:
                    st.markdown("**Example questions you can ask:**")
                    for q in EXAMPLE_QUESTIONS:
                        st.markdown(f"- {q}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
