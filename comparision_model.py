import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Dataset
file_path = '/home/016701321@SJSUAD/summarization/news_summary_more.csv'  # Update with the path to your dataset
df = pd.read_csv(file_path)

# Split dataset into training (80%), validation (10%), and test (10%) sets
train_texts, test_texts, train_summaries, test_summaries = train_test_split(
    df['text'], df['headlines'], test_size=0.2, random_state=42
)
val_texts, test_texts, val_summaries, test_summaries = train_test_split(
    test_texts, test_summaries, test_size=0.5, random_state=42
)

# Convert splits to pandas DataFrames
train_data = pd.DataFrame({'text': train_texts, 'summary': train_summaries})
val_data = pd.DataFrame({'text': val_texts, 'summary': val_summaries})
test_data = pd.DataFrame({'text': test_texts, 'summary': test_summaries})

# Step 2: Load Models and Tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)
bart_model.to(device)

# Step 3: Evaluation Function
def evaluate_model(model, tokenizer, texts, references, max_len=512):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    model.eval()
    for text, reference in zip(texts, references):
        # Tokenize and generate summary
        input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_len, truncation=True).to(device)
        outputs = model.generate(input_ids, max_length=128, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate ROUGE scores
        result = scorer.score(reference, summary)
        scores['rouge1'].append(result['rouge1'].fmeasure)
        scores['rouge2'].append(result['rouge2'].fmeasure)
        scores['rougeL'].append(result['rougeL'].fmeasure)

    # Return average scores
    return {key: sum(values) / len(values) for key, values in scores.items()}

# Step 4: Evaluate Pre-Trained Models
print("Evaluating Pre-Trained T5...")
t5_scores = evaluate_model(t5_model, t5_tokenizer, test_data['text'], test_data['summary'])
print("T5 Pre-Trained Scores:", t5_scores)

print("Evaluating Pre-Trained BART...")
bart_scores = evaluate_model(bart_model, bart_tokenizer, test_data['text'], test_data['summary'])
print("BART Pre-Trained Scores:", bart_scores)

# Step 5: Visualization
def plot_scores(t5_scores, bart_scores, title):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    t5_values = [t5_scores[m] for m in metrics]
    bart_values = [bart_scores[m] for m in metrics]

    x = range(len(metrics))
    plt.bar(x, t5_values, width=0.4, label='T5', align='center')
    plt.bar([i + 0.4 for i in x], bart_values, width=0.4, label='BART', align='center')
    plt.xticks([i + 0.2 for i in x], metrics)
    plt.ylabel("ROUGE Score")
    plt.title(title)
    plt.legend()
    plt.show()

# Plot Pre-Trained Model Results
plot_scores(t5_scores, bart_scores, "Comparison of Pre-Trained T5 and BART Models")
