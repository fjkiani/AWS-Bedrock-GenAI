import streamlit as st
import os
from dotenv import load_dotenv
from main import generate_learning_roadmap, init_neo4j, close_neo4j_connection
import uuid  # For generating unique keys

# Load environment variables
load_dotenv()

# Streamlit Title and Inputs
st.title("The Brain")
st.markdown("""
Welcome to the clustering and catagorizing phase - here all topics are extracted and relationships are made between entities based on similarities 
Type in any learning goal or subject, and we'll generate a personalized learning graph for you!
""")

# User Query Input
user_query = st.text_input("Enter your learning goal (e.g., 'What courses do I need to learn AI?'):")

if st.button("Generate Roadmap"):
    if user_query:
        # Initialize Neo4j Connection
        init_neo4j()
        
        # Generate the roadmap based on user query
        try:
            st.write("Generating your personalized learning roadmap... Please wait.")
            courses, html_content = generate_learning_roadmap(user_query)  # Get HTML content directly
            
            if courses and html_content:
                # Remove the key parameter
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                st.success("Learning roadmap successfully generated!")
            else:
                st.warning("No courses found for your query.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        # Close the Neo4j connection
        close_neo4j_connection()
    else:
        st.warning("Please enter a learning goal.")
