import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from transformers import TrainerCallback
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import zipfile
import tempfile
import shutil
import logging
from huggingface_hub import login, HfFolder
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Tune-IT",
    page_icon="🧠",
    layout="wide"
)

# Define functions for dataset handling
def load_file(uploaded_file):
    """Load and parse uploaded files based on their format."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
        return df
    elif file_extension == 'jsonl':
        data = []
        content = uploaded_file.getvalue().decode('utf-8')
        for line in content.strip().split('\n'):
            if line:  # Skip empty lines
                data.append(json.loads(line))
        return pd.DataFrame(data)
    elif file_extension == 'txt':
        content = uploaded_file.getvalue().decode('utf-8')
        # For text files, create a DataFrame with a single "text" column
        return pd.DataFrame({"text": [content]})
    elif file_extension == 'json':
        data = json.loads(uploaded_file.getvalue().decode('utf-8'))
        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and "data" in data:
            return pd.DataFrame(data["data"])
        else:
            return pd.DataFrame([data])
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

def preprocess_data(df, text_column, label_column=None, test_size=0.2):
    """Preprocess dataframe into training format."""
    # Ensure the text column exists
    if text_column not in df.columns:
        st.error(f"Text column '{text_column}' not found in the dataset.")
        return None, None
    
    # Create dataset dict for Hugging Face datasets
    dataset_dict = {"text": df[text_column].tolist()}
    
    # Add labels if applicable
    if label_column and label_column in df.columns:
        dataset_dict["labels"] = df[label_column].tolist()
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split dataset
    train_test_dataset = dataset.train_test_split(test_size=test_size)
    
    return train_test_dataset["train"], train_test_dataset["test"]

def augment_text_data(dataset, methods=None, factor=2):
    """Apply text augmentation techniques to expand the dataset."""
    if methods is None or not methods:
        return dataset
    
    augmented_texts = []
    augmented_labels = []
    
    for i in range(len(dataset)):
        text = dataset[i]["text"]
        label = dataset[i]["labels"] if "labels" in dataset[i] else None
        
        # Always include the original text
        augmented_texts.append(text)
        if label is not None:
            augmented_labels.append(label)
        
        # Apply selected augmentation methods
        for _ in range(factor - 1):  # -1 because we already have the original
            augmented_text = text
            
            if "Random deletion" in methods:
                words = augmented_text.split()
                # Randomly delete 10-20% of words
                delete_count = max(1, int(len(words) * np.random.uniform(0.1, 0.2)))
                indices_to_delete = np.random.choice(len(words), delete_count, replace=False)
                augmented_text = ' '.join([word for i, word in enumerate(words) if i not in indices_to_delete])
            
            if "Random swapping" in methods and len(augmented_text.split()) > 1:
                words = augmented_text.split()
                # Swap 1-3 pairs of words
                swap_count = min(3, len(words) // 2)
                for _ in range(swap_count):
                    i, j = np.random.choice(len(words), 2, replace=False)
                    words[i], words[j] = words[j], words[i]
                augmented_text = ' '.join(words)
            
            # Note: Proper synonym replacement would require an NLP library like NLTK or spaCy
            # This is a simplified placeholder
            if "Synonym replacement" in methods:
                # In a real implementation, you would use a thesaurus or word embeddings
                # For now, just append a placeholder note
                augmented_text += " [Synonym replacement would modify some words]"
            
            augmented_texts.append(augmented_text)
            if label is not None:
                augmented_labels.append(label)
    
    # Create new augmented dataset
    dataset_dict = {"text": augmented_texts}
    if augmented_labels:
        dataset_dict["labels"] = augmented_labels
    
    return Dataset.from_dict(dataset_dict)

def create_tokenized_dataset(dataset, tokenizer, max_length=512):
    """Tokenize dataset with the model's tokenizer."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def setup_hf_auth():
    """Setup Hugging Face authentication"""
    # Check for existing token
    token_path = Path.home() / '.huggingface' / 'token'
    if token_path.exists():
        return True
    
    # If no token exists, prompt user for token
    st.warning("⚠️ Hugging Face authentication required for gated models")
    token = st.text_input(
        "Enter your Hugging Face token:",
        type="password",
        help="Get your token from https://huggingface.co/settings/tokens"
    )
    
    if token:
        try:
            # Validate and save token
            login(token)
            # Save token to file
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, 'w') as f:
                json.dump({"token": token}, f)
            st.success("✅ Successfully authenticated with Hugging Face!")
            return True
        except Exception as e:
            st.error(f"❌ Authentication failed: {str(e)}")
            return False
    return False

# Add this function after the setup_hf_auth function
def show_auth_popup():
    """Show authentication popup for gated models"""
    st.info("🔒 This model requires Hugging Face authentication")
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        token = st.text_input(
            "Enter your Hugging Face token",
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        auth_button = st.button("Authenticate")
    
    if auth_button and token:
        try:
            login(token)
            # Save token
            token_path = Path.home() / '.huggingface' / 'token'
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, 'w') as f:
                json.dump({"token": token}, f)
            st.success("✅ Successfully authenticated!")
            return True
        except Exception as e:
            st.error(f"❌ Authentication failed: {str(e)}")
            return False
    return None

# Function to load model with download if not available
def load_model_with_download(model_name, local_only=False, custom_token=None):
    """Load model and tokenizer with support for gated models."""
    try:
        # Use custom token if provided, otherwise use the token from HfFolder
        token = custom_token if custom_token else HfFolder.get_token()
        
        # Check if model might be gated
        if not local_only and ('meta-llama' in model_name.lower() or 
                             'llama' in model_name.lower() or 
                             'mistral' in model_name.lower()):
            # Ensure user is authenticated if no custom token
            if not custom_token and not setup_hf_auth():
                raise Exception("Authentication required for gated model access")
        
        if local_only:
            # Only attempt to load from local cache
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                local_files_only=True,
                token=token  # Use the token variable
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                local_files_only=True,
                token=token  # Use the token variable
            )
            return model, tokenizer, "Using locally cached model."
        else:
            try:
                # Try local cache first
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    token=token
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    token=token
                )
                return model, tokenizer, "Using locally cached model."
            except Exception as local_error:
                # Download if not in cache
                logger.info(f"Model not found in cache. Downloading {model_name} from Hugging Face Hub...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=token
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token
                )
                return model, tokenizer, "Model downloaded from Hugging Face Hub."
    except Exception as e:
        raise Exception(f"Failed to load model {model_name}: {str(e)}")
# Custom callback for real-time training monitoring
class StreamlitCallback(TrainerCallback):
    def __init__(self, progress_bar, loss_chart, eval_chart):
        self.progress_bar = progress_bar
        self.loss_chart = loss_chart
        self.eval_chart = eval_chart
        self.training_loss = []
        self.eval_results = []
    
    def on_init_end(self, args, state, control, model=None, **kwargs):
        # Updated to accept model parameter and any additional parameters
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        # Update training loss
        if 'loss' in logs:
            loss = logs['loss']
            step = state.global_step
            self.training_loss.append((step, loss))
            
            # Update loss chart
            steps, losses = zip(*self.training_loss)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=losses, mode='lines', name='Training Loss'))
            fig.update_layout(
                title='Training Loss',
                xaxis_title='Steps',
                yaxis_title='Loss',
                height=400
            )
            self.loss_chart.plotly_chart(fig)
        
        # Update progress bar
        if state.max_steps:
            progress = min(1.0, state.global_step / state.max_steps)
        else:
            progress = min(1.0, state.epoch / args.num_train_epochs)
        
        self.progress_bar.progress(progress)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        
        # Update evaluation metrics
        if 'eval_loss' in metrics:
            step = state.global_step
            eval_loss = metrics['eval_loss']
            self.eval_results.append((step, eval_loss))
            
            # Update eval chart
            steps, losses = zip(*self.eval_results)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=losses, mode='lines+markers', name='Evaluation Loss'))
            fig.update_layout(
                title='Evaluation Loss',
                xaxis_title='Steps',
                yaxis_title='Loss',
                height=400
            )
            self.eval_chart.plotly_chart(fig)

# Function to create and run training with real implementation
def train_model(model, tokenizer, train_dataset, eval_dataset, training_args, callback=None):
    """Set up and run the training process."""
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[callback] if callback else None
    )
    
    # Start training
    trainer.train()
    
    return trainer

def export_model(trainer, model_name, output_dir, formats):
    """Export the model in selected formats."""
    export_files = {}
    
    # Create base directory for the model
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model in Hugging Face format (always included)
    trainer.save_model(output_dir)
    export_files['huggingface'] = output_dir
    
    # Convert to other formats as requested
    if 'pytorch' in formats:
        torch_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(trainer.model.state_dict(), torch_path)
        export_files['pytorch'] = torch_path
    
    if 'gguf' in formats:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            gguf_dir = os.path.join(output_dir, "gguf")
            os.makedirs(gguf_dir, exist_ok=True)
            
            # Configure quantization
            quantize_config = BaseQuantizeConfig(
                bits=4,  # 4-bit quantization
                group_size=128,
                desc_act=False
            )
            
            # Quantize the model
            model_quantized = AutoGPTQForCausalLM.from_pretrained(
                trainer.model,
                quantize_config=quantize_config
            )
            
            # Save the quantized model
            model_quantized.save_pretrained(gguf_dir)
            export_files['gguf'] = gguf_dir
        except ImportError:
            st.warning("GGUF export requires the auto_gptq package. Skipping this format.")
    
    return export_files

def create_zip_from_directory(directory, zip_filename):
    """Create a zip file from a directory."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)
    
    return zip_filename

# Main app layout
def main():
    st.title("🧠 AI Model Fine-tuning Platform")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload Dataset", "Configure Model", "Train Model", "Export Model"])
    
    # Initialize session state
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'train_dataset' not in st.session_state:
        st.session_state.train_dataset = None
    if 'eval_dataset' not in st.session_state:
        st.session_state.eval_dataset = None
    if 'tokenized_train' not in st.session_state:
        st.session_state.tokenized_train = None
    if 'tokenized_eval' not in st.session_state:
        st.session_state.tokenized_eval = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'training_args' not in st.session_state:
        st.session_state.training_args = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'training_callback' not in st.session_state:
        st.session_state.training_callback = None
    
    # Home page
    if page == "Home":
        st.header("Welcome to Tune-IT")
        st.write("""
        This application allows you to fine-tune language models on your own datasets without writing code.
        
        **Follow these steps to fine-tune your model:**
        1. Upload your dataset (CSV, JSONL, or text files)
        2. Configure your model and hyperparameters
        3. Train the model and monitor progress
        4. Export the fine-tuned model in your preferred format
        
        Navigate through the tabs on the sidebar to get started.
        """)
        
        # Display a sample dataset structure
        st.subheader("Example Dataset Format")
        sample_data = {
            "text": [
                "This is a sample text for fine-tuning.",
                "Another example showing how your data should be formatted.",
                "The model will learn from these examples."
            ],
            "labels": [0, 1, 0]  # Optional
        }
        st.dataframe(pd.DataFrame(sample_data))
    # Upload Dataset page
    elif page == "Upload Dataset":
        st.header("Upload and Preprocess Your Dataset")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "jsonl", "json", "txt"])
        
        if uploaded_file is not None:
            try:
                # Load the file
                df = load_file(uploaded_file)
                st.session_state.dataset = df
                
                if df is not None:
                    st.success(f"Successfully loaded dataset with {len(df)} rows.")
                    
                    # Display dataset preview
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head())
                    
                    # Configure preprocessing
                    st.subheader("Configure Preprocessing")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        text_column = st.selectbox(
                            "Select text column",
                            options=df.columns.tolist(),
                            index=0 if 'text' in df.columns else 0
                        )
                    
                    with col2:
                        label_columns = [None] + df.columns.tolist()
                        label_column = st.selectbox(
                            "Select label column (optional)",
                            options=label_columns,
                            index=0
                        )
                    
                    test_size = st.slider(
                        "Test set size (%)",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5
                    ) / 100
                    
                    # Data augmentation options
                    st.subheader("Data Augmentation (Optional)")
                    enable_augmentation = st.checkbox("Enable data augmentation")
                    
                    augmentation_methods = []
                    augmentation_factor = 1
                    
                    if enable_augmentation:
                        augmentation_methods = st.multiselect(
                            "Select augmentation methods",
                            options=["Random deletion", "Random swapping", "Synonym replacement"],
                            default=[]
                        )
                        
                        augmentation_factor = st.slider(
                            "Augmentation factor",
                            min_value=1,
                            max_value=5,
                            value=2,
                            step=1
                        )
                    
                    # Process data button
                    if st.button("Process Dataset"):
                        with st.spinner("Processing dataset..."):
                            # Split into train and test
                            train_dataset, eval_dataset = preprocess_data(
                                df, 
                                text_column, 
                                label_column, 
                                test_size
                            )
                            
                            # If augmentation is enabled, apply it to training set only
                            if enable_augmentation and train_dataset is not None and augmentation_methods:
                                train_dataset = augment_text_data(
                                    train_dataset,
                                    methods=augmentation_methods,
                                    factor=augmentation_factor
                                )
                            
                            if train_dataset is not None and eval_dataset is not None:
                                st.session_state.train_dataset = train_dataset
                                st.session_state.eval_dataset = eval_dataset
                                
                                st.success(f"Dataset processed successfully. Training set: {len(train_dataset)} examples, Evaluation set: {len(eval_dataset)} examples.")
                                
                                # Show examples
                                st.subheader("Training Examples")
                                for i in range(min(3, len(train_dataset))):
                                    st.text_area(f"Example {i+1}", train_dataset[i]["text"], height=100)
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
                st.exception(e)
    
    # Configure Model page
    elif page == "Configure Model":
        st.header("Configure Model and Hyperparameters")
        
        if st.session_state.train_dataset is None:
            st.warning("Please upload and process a dataset first.")
        else:
            # Model selection
            st.subheader("Select Base Model")
            model_options = [
                "gpt2", "gpt2-medium", "gpt2-large",
                "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B",
                "facebook/opt-125m", "facebook/opt-350m",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "meta-llama/Llama-2-7b-hf",  # Add gated models
                "mistralai/Mistral-7B-v0.1",
                "meta-llama/Llama-2-13b-hf"
            ]
            
            model_name = st.selectbox(
                "Choose a pre-trained model",
                options=model_options,
                index=0
            )
            
            # Check if selected model is gated
            is_gated_model = any(name in model_name.lower() for name in ['llama', 'mistral', 'gemma', 'gpt4'])
            
            if is_gated_model:
                token_path = Path.home() / '.huggingface' / 'token'
                if not token_path.exists():
                    auth_status = show_auth_popup()
                    if auth_status is None or auth_status is False:
                        st.warning("⚠️ Please authenticate to use this model")
                        st.stop()
            
            # Add option for custom model path
            custom_model = st.checkbox("Use custom model path")

            if custom_model:
                model_name = st.text_input("Enter custom model path or Hugging Face model ID")
            
                use_custom_token = st.checkbox("Use custom token for gated models")
               
                if use_custom_token:
                     custom_token = st.text_input("Enter your Hugging Face token", type="password")
                else :
                    custom_token = None
 

            
            
            # Option for using only locally cached models
            use_local_only = st.checkbox("Use only locally cached models (no downloads)", value=False)
            
            # Hyperparameter configuration
            st.subheader("Training Hyperparameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                learning_rate = st.number_input(
                    "Learning rate",
                    min_value=1e-7,
                    max_value=1e-2,
                    value=5e-5,
                    format="%.7f",
                    help="Controls how much to adjust the model in response to the error. Smaller values typically lead to slower but more stable training."
                )
                
                batch_size = st.select_slider(
                    "Batch size",
                    options=[1, 2, 4, 8, 16, 32, 64, 128],
                    value=8,
                    help="Number of examples processed at once. Larger batches can be more efficient but require more memory."
                )
                
                max_steps = st.number_input(
                    "Maximum steps",
                    min_value=10,
                    max_value=50000,
                    value=1000,
                    step=100,
                    help="Maximum number of training steps. Set to -1 to train for the specified number of epochs instead."
                )
            
            with col2:
                num_epochs = st.number_input(
                    "Number of epochs",
                    min_value=1,
                    max_value=50,
                    value=3,
                    step=1,
                    help="Number of complete passes through the dataset. Only used if max_steps is set to -1."
                )
                
                warmup_steps = st.number_input(
                    "Warmup steps",
                    min_value=0,
                    max_value=5000,
                    value=500,
                    step=100,
                    help="Number of steps for learning rate warmup. Smaller learning rates are used at the beginning of training."
                )
                
                gradient_accumulation_steps = st.number_input(
                    "Gradient accumulation steps",
                    min_value=1,
                    max_value=32,
                    value=1,
                    step=1,
                    help="Number of steps to accumulate gradients before updating. Useful for training with larger effective batch sizes."
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                weight_decay = st.number_input(
                    "Weight decay",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.01,
                    format="%.4f",
                    help="L2 regularization penalty to prevent overfitting."
                )
                
                fp16 = st.checkbox(
                    "Use mixed precision (FP16)",
                    value=True,
                    help="Enable mixed precision training for faster performance. Requires a compatible GPU."
                )
                
                eval_steps = st.number_input(
                    "Evaluation steps",
                    min_value=50,
                    max_value=5000,
                    value=500,
                    step=50,
                    help="Number of steps between evaluations on the validation set."
                )
                
                save_steps = st.number_input(
                    "Save steps",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Number of steps between saving model checkpoints."
                )
                
                max_length = st.number_input(
                    "Maximum sequence length",
                    min_value=32,
                    max_value=2048,
                    value=512,
                    step=64,
                    help="Maximum length of input sequences. Longer sequences will be truncated."
                )
            
            # Configure model button
            if st.button("Configure Model"):
                with st.spinner("Loading model and tokenizer..."):
                    try:
                        # Load model with download option
                         # Load model with download option and custom token if provided
                        custom_token_value = custom_token if custom_model and use_custom_token else None
                        model, tokenizer, message = load_model_with_download(model_name, local_only=use_local_only, custom_token=custom_token_value)
                        st.success(message)
                        
                        # Add pad token if not present
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        # Tokenize datasets
                        tokenized_train = create_tokenized_dataset(
                            st.session_state.train_dataset, 
                            tokenizer, 
                            max_length
                        )
                        
                        tokenized_eval = create_tokenized_dataset(
                            st.session_state.eval_dataset, 
                            tokenizer, 
                            max_length
                        )
                        
                        # Create training arguments
                        output_dir = f"./results/{model_name.split('/')[-1]}-finetuned"
                        
                        training_args = TrainingArguments(
                            output_dir=output_dir,
                            learning_rate=learning_rate,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            num_train_epochs=num_epochs,
                            max_steps=max_steps if max_steps > 0 else None,
                            warmup_steps=warmup_steps,
                            weight_decay=weight_decay,
                            logging_dir=f"{output_dir}/logs",
                            logging_steps=100,
                            evaluation_strategy="steps",
                            eval_steps=eval_steps,
                            save_steps=save_steps,
                            fp16=fp16,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            save_total_limit=3,
                            load_best_model_at_end=True,
                            report_to="none"  # Disable wandb, etc.
                        )
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.tokenized_train = tokenized_train
                        st.session_state.tokenized_eval = tokenized_eval
                        st.session_state.training_args = training_args
                        
                        st.success(f"Model configured successfully: {model_name}")
                        st.info(f"Training on {len(tokenized_train)} examples, evaluating on {len(tokenized_eval)} examples.")
                        
                        # Display model info
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        
                        st.write(f"Model size: {total_params:,} parameters")
                        st.write(f"Trainable parameters: {trainable_params:,}")
                    except Exception as e:
                        st.error(f"Error configuring model: {str(e)}")
                        st.exception(e)
    
    # Train Model page
    elif page == "Train Model":
        st.header("Train and Monitor Your Model")
        
        if st.session_state.model is None or st.session_state.training_args is None:
            st.warning("Please configure a model first.")
        else:
            # Display configured settings
            st.subheader("Current Configuration")
            if st.session_state.model is not None:
                model_name = st.session_state.model.config._name_or_path
                st.write(f"Model: {model_name}")
            
            if st.session_state.training_args is not None:
                args = st.session_state.training_args
                st.write(f"Learning rate: {args.learning_rate}")
                st.write(f"Batch size: {args.per_device_train_batch_size}")
                st.write(f"Max steps: {args.max_steps}")
                st.write(f"Epochs: {args.num_train_epochs}")
            
            # Training button and monitoring section
            train_button = st.button("Start Training")
            
            if train_button:
                if (st.session_state.model is None or 
                    st.session_state.tokenizer is None or 
                    st.session_state.tokenized_train is None):
                    st.error("Missing required components for training.")
                else:
                    # Create placeholders for monitoring
                    progress_placeholder = st.empty()
                    chart_placeholder = st.empty()
                    eval_placeholder = st.empty()
            
                    # Initialize callback
                    streamlit_callback = StreamlitCallback(
                        progress_placeholder,
                        chart_placeholder,
                        eval_placeholder
                    )
            
                    try:
                        with st.spinner("Training in progress..."):
                            trainer = train_model(
                                st.session_state.model,
                                st.session_state.tokenizer,
                                st.session_state.tokenized_train,
                                st.session_state.tokenized_eval,
                                st.session_state.training_args,
                                streamlit_callback
                            )
                            
                            st.session_state.trainer = trainer
                            st.session_state.training_callback = streamlit_callback
                            
                        st.success("Training completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        st.exception(e)
            
            # Show training results if available
            if 'training_callback' in st.session_state and st.session_state.training_callback is not None:
                callback = st.session_state.training_callback
                
                st.subheader("Training Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot training loss if available
                    if callback.training_loss:
                        steps, losses = zip(*callback.training_loss)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=steps,
                            y=losses,
                            mode='lines',
                            name='Training Loss'
                        ))
                        fig.update_layout(
                            title='Training Loss',
                            xaxis_title='Steps',
                            yaxis_title='Loss',
                            height=400
                        )
                        st.plotly_chart(fig)
                        
                        # Add numerical metrics for training loss
                        st.subheader("Training Loss Metrics")
                        latest_step, latest_loss = callback.training_loss[-1]
                        min_loss = min(losses)
                        min_loss_step = steps[losses.index(min_loss)]
                        
                        metrics_df = pd.DataFrame({
                            "Metric": ["Latest Loss", "Minimum Loss"],
                            "Value": [f"{latest_loss:.6f}", f"{min_loss:.6f}"],
                            "Step": [latest_step, min_loss_step]
                        })
                        st.table(metrics_df)
                
                with col2:
                    # Plot evaluation results if available
                    if callback.eval_results:
                        steps, losses = zip(*callback.eval_results)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=steps,
                            y=losses,
                            mode='lines+markers',
                            name='Evaluation Loss'
                        ))
                        fig.update_layout(
                            title='Evaluation Loss',
                            xaxis_title='Steps',
                            yaxis_title='Loss',
                            height=400
                        )
                        st.plotly_chart(fig)
                        
                        # Add numerical metrics for evaluation loss
                        st.subheader("Evaluation Loss Metrics")
                        latest_step, latest_loss = callback.eval_results[-1]
                        min_loss = min(losses)
                        min_loss_step = steps[losses.index(min_loss)]
                        
                        metrics_df = pd.DataFrame({
                            "Metric": ["Latest Loss", "Minimum Loss"],
                            "Value": [f"{latest_loss:.6f}", f"{min_loss:.6f}"],
                            "Step": [latest_step, min_loss_step]
                        })
                        st.table(metrics_df)
            
    # Export Model page
    elif page == "Export Model":
        st.header("Export Your Fine-tuned Model")
        
        if st.session_state.trainer is None:
            st.warning("Please train a model first.")
        else:
            st.subheader("Export Configuration")
            
            # Choose export formats
            export_formats = st.multiselect(
                "Select export formats",
                options=["huggingface", "pytorch", "gguf"],
                default=["huggingface"]
            )
            
            # Model name
            if st.session_state.model is not None:
                default_name = st.session_state.model.config._name_or_path.split('/')[-1] + "-finetuned"
            else:
                default_name = "finetuned-model"
            
            model_name = st.text_input("Model name", value=default_name)
            
            # Export button
            if st.button("Export Model"):
                with st.spinner("Exporting model..."):
                    try:
                        # Create temporary directory for export
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            output_dir = os.path.join(tmp_dir, model_name)
                            
                            # Export model in selected formats
                            export_files = export_model(
                                st.session_state.trainer,
                                model_name,
                                output_dir,
                                export_formats
                            )
                            
                            # Create a zip file for download
                            zip_path = os.path.join(tmp_dir, f"{model_name}.zip")
                            create_zip_from_directory(output_dir, zip_path)
                            
                            # Display success message with details
                            st.success(f"Model exported successfully in the following formats: {', '.join(export_formats)}")
                            
                            # Create download button
                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    label="Download Model",
                                    data=f,
                                    file_name=f"{model_name}.zip",
                                    mime="application/zip"
                                )
                            
                            # Display additional information
                            st.subheader("Model Information")
                            st.json({
                                "Model Name": model_name,
                                "Base Model": st.session_state.model.config._name_or_path,
                                "Export Formats": export_formats,
                                "Size": "Size info will be available after download"
                            })
                            
                            # Display usage instructions
                            st.subheader("How to Use Your Model")
                            usage_code = """
# Load your fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/extracted/model")
model = AutoModelForCausalLM.from_pretrained("path/to/extracted/model")

# Example usage
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
"""
                            st.code(usage_code, language="python")
                            
                    except Exception as e:
                        st.error(f"Error during export: {str(e)}")
                        st.exception(e)

if __name__ == "__main__":
    main()
