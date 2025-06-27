import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading
import time

# Show title and description.
st.title("💬 Local Chatbot")
st.write(
    "This chatbot uses a local Phi-4 model to generate responses. "
    "The model runs entirely on your machine - no API keys required!"
)

# Load model and tokenizer (cached in session state)
@st.cache_resource
def load_model():
    model_id = "microsoft/Phi-4-mini-instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Generator function for streaming
def generate_streaming_response(model, tokenizer, messages):
    # Convert messages to prompt
    conversation = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"
    conversation += "Assistant:"
    
    # Tokenize with smaller context
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=512)  # Reduced from 2048
    
    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generation parameters - optimized for speed
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 50,          # Reduced from 150
        "temperature": 0.8,
        "do_sample": True,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Start generation in separate thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they come
    for token in streamer:
        yield token
        time.sleep(0.01)  # Small delay for better visual effect

# Load model once
with st.spinner("Loading Phi-4 model... (this may take a moment on first run)"):
    model, tokenizer = load_model()

st.success("✅ Model loaded successfully!")

# Create a session state variable to store the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the local Phi-4 model with streaming.
    with st.chat_message("assistant"):
        stream = generate_streaming_response(model, tokenizer, st.session_state.messages)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})