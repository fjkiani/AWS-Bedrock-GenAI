# chatbot_frontend.py

import streamlit as st
import chatbot_backend as demo  # Import your Chatbot file as demo

# Set Title for Chatbot
st.title("Hi, This is JediLabs.org bot :sunglasses:")

# LangChain memory to the session cache - Session State
if 'memory' not in st.session_state: 
    st.session_state.memory = demo.demo_memory()  # Initialize the memory

# Add the UI chat history to the session cache - Session State
if 'chat_history' not in st.session_state:  # See if the chat history hasn't been created yet
    st.session_state.chat_history = []  # Initialize the chat history

# Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 

# Enter the details for chatbot input box 
input_text = st.chat_input("Powered by Bedrock and Claude")  # Display a chat input box
if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({"role": "user", "text": input_text}) 

    chat_response = demo.demo_conversation(input_text=input_text, memory=st.session_state.memory)  # Call the model through the supporting library
    
    with st.chat_message("assistant"): 
        st.markdown(chat_response) 
    
    st.session_state.chat_history.append({"role": "assistant", "text":chat_response}) 
