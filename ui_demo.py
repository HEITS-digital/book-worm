import gradio as gr
import time
from itertools import chain


def echo(message, history):
    response = f"I'm just repeating what you're typing\n Your message: {message}.\n"
    for i in range(len(response)):
        time.sleep(0.05)
        yield response[: i+1]

def i_am_a_bot():
    time.sleep(1)
    response = "Oh ... here's the book"
    for i in range(len(response)):
        time.sleep(0.05)
        yield response[: i+1]

def chainer(message, history):
    generator3 = chain(echo(message, history), i_am_a_bot())
    for i in generator3:
        yield i

demo = gr.ChatInterface(
    chainer,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me about a book", container=False, scale=7),
    title="Book Worm",
    description="Your AI librarian for classical literature",
    examples=[
        "Do you know the book Guliver's Travels?", 
        "What was the name of the first island that Guliver got to?"
    ],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch(share=True)
