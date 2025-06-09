import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json
import numpy as np
from collections import Counter
import math

MODEL_DIR = "gpt2"  # Using the base GPT-2 model from Hugging Face
DATASET_PATH = "./data/waec_qa_dataset.jsonl"

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

def main():
    st.title("PrepMate: WASSCE Exam Preparation Assistant")
    st.write("Ace your WASSCE prep — ask questions in Math, English, Science, or more, and get answers with easy-to-understand explanations.")

    # Load model and dataset
    tokenizer, model = load_model()
    dataset_questions = load_dataset()

    user_input = st.text_input("Your question:")

    if st.button("Get Answer"):
        if user_input.strip():
            # Calculate similarity with dataset
            similarities = [calculate_similarity(user_input, q) for q in dataset_questions]
            similarity_score = max(similarities) if similarities else 0
            
            # Generate response
            prompt = f"Question: {user_input}\nAnswer:"
            with st.spinner("Generating answer..."):
                answer, confidence = generate_response(tokenizer, model, prompt)
            
            # Display response with confidence indicators
            st.markdown(f"**Answer:** {answer}")
            
            # Show confidence level
            confidence_level = min(similarity_score, confidence)
            if confidence_level < 0.3:
                st.warning("⚠️ This question appears to be outside our trained domain. The answer may not be accurate.")
                st.info("Consider rephrasing your question or asking about a different topic.")
            elif confidence_level < 0.6:
                st.info("ℹ️ Moderate confidence in this answer. Please verify the information.")
            
            # Show similar questions if available
            similar_questions = find_similar_questions(user_input, dataset_questions)
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
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
