import time
import gradio as gr

from book_worm import BookWorm


def update_env_vars(env_file_path: str = None):
    import os
    from dotenv import dotenv_values

    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}


if __name__ == "__main__":
    update_env_vars(".env")

    def echo(message, history):
        # need to find a better way to instantiate this, BC it cannot be declared globally :(
        chat = BookWorm(history)
        response = chat.ask_bookworm(message)
        output = response["output"]
        for i in range(len(output)):
            time.sleep(0.02)
            yield output[: i + 1]

    demo = gr.ChatInterface(
        echo,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me about a book", container=False, scale=7),
        title="Book Worm",
        description="Your AI librarian for classical literature",
        examples=[
            ["What would be a good book about pirates and adventure?"],
            ["I started reading a lot of Jonathan Swift lately. Can you recommend any similar authors?"],
            ["Are there any books by Mark Twain in bookworm?"],
            ["""Is there any horse in "A Horse's Tale" by Mark Twain?"""],
            ["""Who is Buffalo Bill from "A Horse's Tale" by Mark Twain?"""]
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
        concurrency_limit=10,
    )

    demo.launch(share=True)
