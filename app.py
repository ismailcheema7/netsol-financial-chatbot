import gradio as gr
from functions import create_interface, running_agent  # make sure running_agent is also imported
from fastapi import FastAPI
import uvicorn
from gradio.routes import App as GradioApp
from pydantic import BaseModel
from rag_pipeline import rag_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI is running, go to /docs or /chat"}

demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/gradio")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
