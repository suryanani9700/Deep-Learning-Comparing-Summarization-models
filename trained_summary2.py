import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

model_path = './my_bart_model'
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cpu")
model.to(device)
model.eval()

def summarize_text(input_text, max_length=150, min_length=50, num_beams=6):
    inputs = tokenizer("summarize: " + input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,        
        min_length=min_length,        
        num_beams=num_beams,        
        length_penalty=1.2,          
        early_stopping=False         
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def evaluate_summary(generated_summary, ground_truth):
    bleu_score = sentence_bleu([ground_truth.split()], generated_summary.split())
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(ground_truth, generated_summary)
    
    return bleu_score, rouge_scores

st.title("Text Summarization with Evaluation Metrics")

st.write("Enter text to summarize and provide the ground truth summary to evaluate the results.")

input_text = st.text_area("Enter the text to summarize", height=200)
ground_truth = st.text_area("Enter the ground truth summary", height=100)

if st.button("Summarize and Evaluate"):
    if input_text.strip() and ground_truth.strip():
        with st.spinner("Generating summary..."):
            generated_summary = summarize_text(input_text)

        with st.spinner("Evaluating summary..."):
            bleu_score, rouge_scores = evaluate_summary(generated_summary, ground_truth)

        st.subheader("Generated Summary")
        st.write(generated_summary)

        st.subheader("Evaluation Metrics")
        st.write(f"**BLEU Score:** {bleu_score:.4f}")
        st.write("**ROUGE Scores:**")
        st.write(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
        st.write(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
        st.write(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
    else:
        st.error("Please enter both the text to summarize and the ground truth summary!")
