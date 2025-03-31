import transformers
from transformers import pipeline
import torch
import streamlit as st
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.title("Sentiment Analyzer")

# Define your label names (in the correct order)
label_names = ["neutral", "positive", "mixed", "sarcastic", "negative", "ironic"]

# Define your custom thresholds for each label
custom_thresholds = {
    'neutral': 0.39,
    'positive': 0.509,
    'mixed': 0.22,
    'sarcastic': 0.29,
    'negative': 0.428,
    'ironic': 0.16
}

tokenizer = AutoTokenizer.from_pretrained("OmarBrookes/my-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("OmarBrookes/my-sentiment-analysis")

# Update your model config so that labels are human-readable.
model.config.id2label = {i: label for i, label in enumerate(label_names)}
model.config.label2id = {label: i for i, label in enumerate(label_names)}

# Create the pipeline. Using "sigmoid" with top_k=None returns probabilities for all labels.
pipe = pipeline(
    task="text-classification",
    model="OmarBrookes/my-sentiment-analysis",
    tokenizer=tokenizer,
    function_to_apply="sigmoid",  # For multi-label, use sigmoid activation
    top_k=None,                   # Return scores for all labels
    device=0                      # Use GPU (device 0) if available
)

# Prediction function with custom thresholds.
def predict_with_custom_thresholds(text, thresholds):
    # Get raw output from the pipeline.
    output = pipe(text)
    # Check if output is a nested list; if so, flatten it.
    if isinstance(output[0], list):
        flat_output = [item for sublist in output for item in sublist]
    else:
        flat_output = output

    # Apply custom thresholds: only keep labels where score meets/exceeds the threshold.
    final_labels = [entry['label'] for entry in flat_output if entry['score'] >= thresholds[entry[0]['label']]]
    return  final_labels,flat_output
# final_labels,

input=st.text_input("Input Text Here")
if st.button("Analyze"):
  final_labels , raw_output= predict_with_custom_thresholds(input, 3)
  st.write(final_labels)
  st.write(raw_output)
  # for entry in raw_output:
  #       st.write(f"Label: {entry['label']}, Score: {entry['score']:.4f}")
  #       st.write("\nFinal Predicted Labels (after applying custom thresholds):")
  #       st.write(final_labels)
    
# raw_output


# # Interactive loop for predictions
# def interactive_prediction():
#     while True:
#         text = input("Enter your text for sentiment analysis: ")
#         final_labels, raw_output = predict_with_custom_thresholds(text, custom_thresholds)
#         st.write("\nRaw Prediction Scores:")
#         for entry in raw_output:
#             print(f"Label: {entry['label']}, Score: {entry['score']:.4f}")
#         print("\nFinal Predicted Labels (after applying custom thresholds):")
#         st.write(final_labels)
#         cont = input("\nWould you like to analyze another text? (y/n): ")
#         if cont.lower() != "y":
#             break

# # Run the interactive prediction loop
# interactive_prediction()
