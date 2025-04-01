import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
MODEL_PATH = "OmarBrookes/my-sentiment-analysis"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABELS = ['neutral', 'positive', 'mixed', 'sarcastic', 'negative', 'ironic']

custom_thresholds = {
    'neutral': 0.39,
    'positive': 0.509,
    'mixed': 0.22,
    'sarcastic': 0.29,
    'negative': 0.428,
    'ironic': 0.16
}

if "review_count" not in st.session_state:
    st.session_state.review_count = 1 

st.title("Sentiment Analysis")
st.write("Enter a text and get sentiment predictions.")

user_input = st.text_area("Enter your text here:")
expected_output = st.text_input("Expected Sentiments (comma-separated, optional):")

if st.button("Analyze Sentiment") and user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]  # Convert logits to probabilities

    prob_dict = {label: prob for label, prob in zip(LABELS, probs)}

    final_labels = [label for label in LABELS if prob_dict[label] >= custom_thresholds[label]]
    expected_labels = [label.strip() for label in expected_output.split(",")] if expected_output else ["N/A"]
    st.write(f"### 🔹 Review #{st.session_state.review_count}")
    st.write(f"📝 *Text:* {user_input}")

    st.write(f"🎯 *Expected:* {expected_labels}")
    st.write(f"✅ *Predicted:* {final_labels}")
    st.write("📊 *Probabilities:*")
    for label, prob in prob_dict.items():
        st.write(f"🔹 {label}: {prob:.4f}")

    st.session_state.review_count += 1

    st.write("---")
