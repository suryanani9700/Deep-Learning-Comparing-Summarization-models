import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import json

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.eval()

def summarize(text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
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

st.title("T5 Text Summarization with Evaluation Metrics")

text = st.text_area("Text to summarize")
reference = st.text_area("Reference summary (optional)")

if st.button('Generate Summary'):
    summary = summarize(text)
    st.subheader("Generated Summary")
    st.text_area("Generated Summary", value=summary, height=200)
    
    if reference:
        st.subheader("Evaluation Metrics")
        scores = calculate_rouge_scores(summary, reference)
        st.json(scores)
