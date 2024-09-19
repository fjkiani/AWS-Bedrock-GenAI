# The below frontend code is provided by AWS and Streamlit. I have only modified it to make it look attractive.
import streamlit as st 
import rag_backend as demo 

st.set_page_config(page_title="Retrieval Augmented Generation") ### Modify Heading

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">HR Q & A with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True) ### Modify Title

if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Wait for magic...All beautiful things in life take time :-)"): ###spinner message
        st.session_state.vector_index = demo.hr_index() ### Your Index Function name from Backend File

input_text = st.text_area("Input text", label_visibility="collapsed") 
go_button = st.button("📌Grow your business 5x with JediLabs.org", type="primary") ### Button Name

if go_button: 
    
    with st.spinner("📢"): ### Spinner message
        response_content = demo.hr_rag_response(index=st.session_state.vector_index, question=input_text) ### replace with RAG Function from backend file
        st.write(response_content) 