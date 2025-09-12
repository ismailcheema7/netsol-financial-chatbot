from typing import Sequence, Annotated, TypedDict
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
import gradio as gr
from rag_pipeline import rag_agent
import json

def chat_with_agent(message, history):
    """Function to handle chat interactions with the RAG agent"""
    try:
        messages = []
        for user_msg, bot_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))

        messages.append(HumanMessage(content=message)) 
        print("🔁 Message history sent to RAG agent:")
        for m in messages:
            print("-", m.content)

        result = rag_agent.invoke({"messages": messages})
        response = result['messages'][-1].content
        return response
    
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}"

custom_css = """
.gradio-container {
    background-color: #1a1d3a !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Chat interface dark base */
.dark {
    background-color: #1a1d3a !important;
}
/* Message styling */
.message {
    font-size: 1rem !important;
    background-color: transparent !important;
    color: #ffffff !important;
    box-shadow: none !important;
    border: none !important;
}

/* User message */
.message.user {
    background-color: #3a3d7a !important;
    border-radius: 12px !important;
    padding: 8px 12px;
    margin: 6px 0;
}

/* Bot message */
.message.ai {
    background-color: #2f3266 !important;
    border-radius: 12px !important;
    padding: 8px 12px;
    margin: 6px 0;
}

/* Input box */
textarea {
    background-color: #2a2d5a !important;
    color: #ffffff !important;
    border: 1px solid #4a4d8a !important;
    border-radius: 8px !important;
}

/* Button */
button {
    background-color: #4a4d8a !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #5b5eb0 !important;
}

/* Logo */
#logo {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 20px;
    padding: 10px;
    background: linear-gradient(45deg, #4a4d8a, #2a2d5a);
    border-radius: 10px;
}
"""


# Create the interface
def create_interface():
    with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome(), title="NetSol Financial Report Chatbot") as demo:

        # Header with logo
        gr.HTML("""
        <div id="logo">
            📊 NetSol Financial Assistant
            <br>
            <small>Your AI-powered financial analyst for NetSol Technologies</small>
        </div>
        """)


        chatbot = gr.ChatInterface(
                fn=chat_with_agent,
                title="",
                description="Ask me anything about NetSol's 2024 financial report! I can search through the documents and even check the web if needed.",
                examples=[
                    "What is NetSol's revenue for 2024?",
                    "Tell me about NetSol's profitability",
                    "What are the company's main expenses?",
                    "How is NetSol's cash flow?",
                    "What are the key financial highlights?"
                ],
                cache_examples=False,
                submit_btn="Send 📤",
                css= """{
                        background-color: #2a2d5a !important;
                        border-radius: 10px !important;
                        box-shadow: 0 0 12px rgba(0, 0, 0, 0.3);
                        height: 600px !important;
                        max-height: 600px !important;
                        width: 100% !important;
                        overflow-y: auto !important;
                        }
                        """)

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 10px; color: #888;">
            <small>💡 Tip: I can search NetSol's financial documents and provide web-based information when needed!</small>
        </div>
        """)

    return demo

def running_agent():
    print("\n=== RAG and Tavily AGENT===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = [HumanMessage(content=user_input)]

        try:
            result = rag_agent.invoke({"messages": messages})
            print("\n=== ANSWER ===")
            print(result['messages'][-1].content)
        except Exception as e:
            print(f"Error: {e}")