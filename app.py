import streamlit as st
from groq_client import GroqClient
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return GroqClient()

groq_client = get_groq_client()

# App header
st.title("Groq Chatbot")
st.markdown("Chat with Groq's powerful language models")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare messages for API call
        messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        # Stream the response
        for response_chunk in groq_client.generate_response(messages_for_api):
            full_response += response_chunk
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)  # Small delay for better streaming visualization
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
