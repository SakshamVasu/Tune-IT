import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
def setup_page():
    st.set_page_config(
        page_title="Tune-IT",
        page_icon="💎",
        layout="wide"
    )
    st.title("Tune-IT💎")
    st.markdown("""
    This application allows you to easily fine-tune open-source language models on your own datasets.
    Simply upload your data, configure the training parameters, and start fine-tuning!
    """)

# Sidebar for authentication and model selection
def sidebar_components():
    with st.sidebar:
        st.header("Authentication")
        hf_token = st.text_input("Hugging Face Token", type="password")
        
        token_valid = False
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                st.success("Token verified successfully!")
                token_valid = True
            except Exception as e:
                st.error(f"Token validation failed: {e}")
        else:
            st.warning("Please enter your Hugging Face token")

        st.header("Model Selection")
        model_options = [
            "google/gemma-2b",
            "google/gemma-2b-it",
            "google/gemma-7b", 
            "google/gemma-7b-it"
        ]
        selected_model = st.selectbox("Select Model", model_options)
    
    return hf_token, token_valid, selected_model

# Data upload UI
def data_upload_section():
    st.header("1. Upload Your Dataset")
    data_format = st.radio("Select data format:", ["CSV", "JSON/JSONL", "Text Files (TXT)"])
    uploaded_file = st.file_uploader(
        f"Upload your {data_format} file",
        type=["csv", "json", "jsonl", "txt"] if data_format == "Text Files (TXT)" else data_format.lower()
    )
    return data_format, uploaded_file

# Training configuration UI
def training_config_section():
    st.header("2. Configure Fine-tuning")
    with st.expander("Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            training_method = st.radio("Training Method", ["LoRA (recommended)", "Full Fine-tuning"])
            epochs = st.slider("Number of epochs", min_value=1, max_value=10, value=3)
            batch_size = st.slider("Batch size", min_value=1, max_value=32, value=4)
            
            st.subheader("Learning Rate")
            learning_rate_input = st.text_input(
                "Enter Learning Rate (e.g., 0.00001, 0.0001, etc.)",
                value="0.00005",
                help="Enter a value between 1e-8 and 1e-2"
            )
            try:
                learning_rate = float(learning_rate_input)
                if learning_rate <= 0:
                    st.error("Learning rate must be greater than 0. Using default value 5e-5.")
                    learning_rate = 5e-5
                elif learning_rate > 0.01:
                    st.warning("Warning: High learning rate may cause training instability.")
            except ValueError:
                st.error("Invalid learning rate format. Using default value 5e-5.")
                learning_rate = 5e-5
            st.info(f"Selected Learning Rate: {learning_rate:.8f}")
        
        with col2:
            max_length = st.slider("Max sequence length", min_value=128, max_value=2048, value=512)
            validation_split = st.slider("Validation split", min_value=0.0, max_value=0.3, value=0.1)
            
            if training_method == "LoRA (recommended)":
                lora_rank = st.slider("LoRA rank (r)", min_value=1, max_value=64, value=8)
                lora_alpha = st.slider("LoRA alpha", min_value=1, max_value=64, value=16)
                lora_dropout = st.slider("LoRA dropout", min_value=0.0, max_value=0.5, value=0.05, step=0.05)
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                st.info(f"Using target modules: {', '.join(lora_target_modules)}")
            else:
                lora_rank, lora_alpha, lora_dropout, lora_target_modules = None, None, None, None
        
        return (training_method, epochs, batch_size, learning_rate, max_length, validation_split,
                lora_rank, lora_alpha, lora_dropout, lora_target_modules)

# Plot training progress
def plot_training_progress(logs, placeholder):
    if logs:
        fig, ax = plt.subplots(figsize=(6, 4))
        epochs = [log.get("epoch", idx) for idx, log in enumerate(logs) if "loss" in log]
        train_loss = [log.get("loss") for log in logs if "loss" in log]
        eval_epochs = [log.get("epoch") for log in logs if "eval_loss" in log]
        eval_loss = [log.get("eval_loss") for log in logs if "eval_loss" in log]
        
        if train_loss:
            ax.plot(epochs, train_loss, label="Training Loss", marker="o")
        if eval_loss:
            ax.plot(eval_epochs, eval_loss, label="Validation Loss", marker="x")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        placeholder.pyplot(fig)

# Display metrics
def display_metrics(logs, placeholder):
    if logs:
        latest = logs[-1]
        metrics_text = "### Training Metrics\n\n"
        for key, value in latest.items():
            if isinstance(value, (int, float)):
                metrics_text += f"{key}: {value:.4f}\n\n"
            else:
                metrics_text += f"{key}: {value}\n\n"
        placeholder.markdown(metrics_text)

# CSS styling
def apply_css():
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .stProgress .st-bd {
            height: 20px;
        }
        h1, h2, h3 {
            color: #357736;
        }
        .stAlert {
            background-color: #f1f9f2;
        }
    </style>
    """, unsafe_allow_html=True)