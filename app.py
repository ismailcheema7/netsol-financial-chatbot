import gradio as gr
from functions import create_interface, running_agent  # make sure running_agent is also imported
from fastapi import FastAPI
import uvicorn
from gradio.routes import App as GradioApp
from pydantic import BaseModel
from rag_pipeline import rag_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI is running, go to /docs or /chat"}

demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/gradio")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    history: List[Dict[str, str]]  # [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    question: str  # new user question

# Endpoint
@app.post("/chat")
def post_on_fastapi(req: ChatRequest):
    try:
        # Convert chat history to LangChain messages
        messages = []
        for msg in req.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=req.question))

        # RAG invocation
        result = rag_agent.invoke({"messages": messages})

        return {"response": result["messages"][-1].content}
    
    except Exception as e:
        return {"error": str(e)}