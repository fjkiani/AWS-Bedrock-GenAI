import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Load environment variables
load_dotenv()

# Set up constants
AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize components
chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo")

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# Define chunking strategy
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# Define Streamlit interface
st.title("RAG with Neo4j and LangChain")
st.sidebar.header("Configuration")
query = st.text_input("Enter your query", "How did the Roman Empire fall?")

# Load Wikipedia documents (for demonstration)
if st.button("Load Wikipedia Data"):
    raw_documents = WikipediaLoader(query="The Roman Empire").load()
    documents = text_splitter.split_documents(raw_documents[:3])
    st.write(f"Loaded and split {len(documents)} documents.")
    llm_transformer = LLMGraphTransformer(llm=chat)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    res = kg.add_graph_documents(
        graph_documents,
        include_source=True,
        baseEntityLabel=True,
    )
    st.write("Documents added to Neo4j graph.")

# Define retrieval function
def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data

# Define structured retriever
def structured_retriever(question: str) -> str:
    entities = entity_chain.invoke({"question": question})
    result = ""
    for entity in entities.names:
        response = kg.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response])
    return result

# Handle query submission
if st.button("Submit Query"):
    with st.spinner("Retrieving data..."):
        result = retriever(query)
        st.write("### Results")
        st.write(result)

# Launch the app with `streamlit run app.py`
