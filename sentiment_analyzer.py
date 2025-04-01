import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
MODEL_PATH = "OmarBrookes/my-sentiment-analysis"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
LABELS = {
    'neutral': "😐 Neutral",
    'positive': "😊 Positive",
    'mixed': "😕 Mixed",
    'sarcastic': "😏 Sarcastic",
    'negative': "😠 Negative",
    'ironic': "🙃 Ironic"
}

custom_thresholds = {
    'neutral': 0.34,
    'positive': 0.509,
    'mixed': 0.28,
    'sarcastic': 0.31,
    'negative': 0.428,
    'ironic': 0.28
}

test_reviews = [
    ("I absolutely love this product!", ['positive']),
    ("Not great, not terrible. Just okay.", ['neutral']),
    ("Oh wow, another amazing day at work...", ['sarcastic']),
]

if 'review_count' not in st.session_state:
    st.session_state.review_count = 0

st.title("Sentiment Analysis with RoBERTa")
st.write("Enter a text and get sentiment predictions.")

user_input = st.text_area("Enter your text here:")

expected_label = st.selectbox("Expected Sentiment (from dataset):", [str(lbl) for _, lbl in test_reviews])

if st.button("Analyze Sentiment") and user_input:
    st.session_state.review_count += 1  
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    predicted_labels = [LABELS[label] for label, prob in zip(LABELS.keys(), probs) if prob >= custom_thresholds[label]]
    
    st.subheader(f"Review #{st.session_state.review_count}")
    st.write(f"📝 Input:** {user_input}")
    st.write(f"🎯 *Expected:* {expected_label}")
    st.write(f"✅ *Predicted:* {', '.join(predicted_labels) if predicted_labels else 'No strong sentiment detected'}")
    
    st.write("📊 *Confidence Scores:*")
    for label, prob in zip(LABELS.keys(), probs):
        st.write(f"{LABELS[label]}: *{prob:.4f}*")
