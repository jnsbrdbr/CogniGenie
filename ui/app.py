# app.py
import gradio as gr
import requests

API_URL = "http://localhost:8000"
token = None

def login(username, password):
    global token
    res = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
    if res.status_code == 200:
        token = res.json()["access_token"]
        return "Login successful! Go to Chat tab."
    return f"Login failed! {res.text}"

def ask_backend(message, history, with_sources=False):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    endpoint = "/chat_with_sources" if with_sources else "/chat"
    res = requests.post(f"{API_URL}{endpoint}", json={"question": message}, headers=headers)
    if res.status_code != 200:
        return f"Error: {res.text}"
    data = res.json()
    if with_sources and "sources" in data:
        src = "\n".join(f"- {s['doc_id']} (score={s['score']})" for s in data["sources"])
        return f"{data['answer']}\n\n---\nSources:\n{src}"
    return data["answer"]

with gr.Blocks() as demo:
    with gr.Tab("Login"):
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox()
        login_btn.click(login, [username, password], login_output)

    with gr.Tab("Chat"):
        gr.Markdown("**Chat (no sources)**")
        gr.ChatInterface(fn=lambda m, h: ask_backend(m, h, with_sources=False), title="Secure RAG Chatbot")

    with gr.Tab("Chat + Sources"):
        gr.Markdown("**Chat with retrieved sources**")
        gr.ChatInterface(fn=lambda m, h: ask_backend(m, h, with_sources=True), title="Secure RAG Chatbot (Sources)")

demo.launch(server_name="0.0.0.0", server_port=7860)
