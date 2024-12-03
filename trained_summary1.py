import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from rouge_score import rouge_scorer

model_path = '/home/016701321@SJSUAD/summarization/my_t5_model'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")  
def summarize_text(input_text):
    model.eval()
    input_ids = tokenizer(
        "summarize: " + input_text,
        return_tensors="pt",
        max_length=512, 
        truncation=True
    ).input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=200, 
            min_length=50,  
            num_beams=6,     
            length_penalty=2.0,  
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_rouge_scores(summary, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, summary)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4)
    }

st.title('Text Summarization and Evaluation Tool')

col1, col2 = st.columns(2)
with col1:
    input_text = st.text_area("Enter the text to summarize:", height=300)
with col2:
    ground_truth_text = st.text_area("Enter the ground truth summary:", height=300)

if st.button("Summarize"):
    if input_text.strip() and ground_truth_text.strip():
        try:
            summary = summarize_text(input_text)
            st.subheader("Generated Summary:")
            st.write(summary)

            rouge_scores = calculate_rouge_scores(summary, ground_truth_text)
            st.subheader("ROUGE Scores:")
            st.json(rouge_scores)
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
    else:
        st.warning("Please enter both the text to summarize and the ground truth summary.")
