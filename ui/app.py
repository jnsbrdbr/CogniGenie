import gradio as gr
import requests

# Updated function to handle conversation history
def chat_with_backend(message, history):
    res = requests.post("http://localhost:8000/chat", json={"question": message})
    answer = res.json()["answer"]
    return answer

gr.ChatInterface(
    fn=chat_with_backend,
    title="Local RAG Chatbot"
).launch()