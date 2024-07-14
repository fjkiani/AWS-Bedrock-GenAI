# chatbot_backend.py

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import ChatBedrock

# Function for invoking model-client connection with Bedrock
def demo_chatbot():
    demo_llm = ChatBedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        model_kwargs={
            "max_tokens": 300,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:"]
        }
    )
    return demo_llm

# Function for ConversationSummaryBufferMemory
def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=300)
    return memory

# Function for Conversation Chain - Input text + Memory
def demo_conversation(input_text, memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    
    # Chat response using invoke (Prompt template)
    chat_reply = llm_conversation.invoke(input_text)
    return chat_reply['response']
