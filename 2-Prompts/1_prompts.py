from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch
import streamlit as st

load_dotenv()

# Page config
st.set_page_config(page_title="BART Text Summarizer", layout="wide")

# Title and description
st.title("📝 BART-Powered Text Summarizer")
st.write("Paste your long text below, and get a concise summary using facebook/bart-large-cnn!")

# Load model (done once at startup)
@st.cache_resource
def load_summarizer():
    """Load the BART summarization pipeline."""
    device = 0 if torch.cuda.is_available() else -1  # GPU if available
    pipe = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device,
        temperature=1.5,
        max_new_tokens=130,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,     # Deterministic output
        truncation=True      # Handle long inputs
    )
    return HuggingFacePipeline(pipeline=pipe)

# Initialize the model
try:
    model = load_summarizer()
    # st.success("Model loaded successfully! (This may take a minute on first run.)")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User input
user_input = st.text_area(
    "Enter text to summarize:",
    height=200,
    placeholder="Paste your article, paragraph, or any long text here..."
)

# Summarize button
if st.button("Generate Summary", type="primary"):
    if user_input.strip():
        with st.spinner("Summarizing... This may take 10-30 seconds."):
            try:
                # Invoke the pipeline with plain text
                summary = model.invoke(user_input)
                # st.success("Summary generated!")
                st.write("### Summary")
                st.write(summary)
                
                # Optional: Show original length vs. summary length
                orig_len = len(user_input.split())
                sum_len = len(summary.split())
                st.metric("Original Words", orig_len)
                st.metric("Summary Words", sum_len)
                st.info(f"Compression: {sum_len / orig_len * 100:.1f}% of original length")
                
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                st.info("Tips: Keep input under 1024 tokens (~800 words) for best results.")
    else:
        st.warning("Please enter some text to summarize.")

# Sidebar tips
# with st.sidebar:
#     st.header("Tips")
#     st.write("- BART works best on news/articles (factual text).")
#     st.write("- For longer texts, split into chunks if needed.")
#     st.write("- GPU acceleration: Enable if you have CUDA setup.")
#     st.write("\n**Dependencies:** `pip install streamlit langchain-huggingface transformers torch`")