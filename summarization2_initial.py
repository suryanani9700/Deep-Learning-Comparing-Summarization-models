import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
import json

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.eval()

def summarize(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_rouge_scores(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    formatted_scores = {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 4),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 4)
    }
    return formatted_scores

st.title("BART Text Summarization with Evaluation Metrics")

text = st.text_area("Text to summarize", height=200)
reference = st.text_area("Reference summary (optional)", height=200)

if st.button('Generate Summary'):
    if not text.strip():
        st.warning("Please enter text to summarize.")
    else:
        summary = summarize(text)
        st.subheader("Generated Summary")
        st.text_area("Generated Summary", value=summary, height=200)
        
        if reference.strip():
            st.subheader("Evaluation Metrics")
            scores = calculate_rouge_scores(summary, reference)
            st.json(scores)
        else:
            st.info("Provide a reference summary to calculate evaluation metrics.")
