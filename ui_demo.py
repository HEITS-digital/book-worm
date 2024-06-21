import time
import gradio as gr

from bookwom import BookWorm


# def echo(message, history):
#     response = f"Hmmm ... let me look in the library for that ..."
#     for i in range(len(response)):
#         time.sleep(0.02)
#         yield response[: i+1]


# def i_am_a_bot(message):
#     response = chat.parse_message(message)
#     yield response


# def chainer(message, history):
#     generator3 = chain(echo(message, history), i_am_a_bot(message))
#     for i in generator3:
#         yield i


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
        chat = BookWorm()
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
            "Do you know the book Guliver's Travels?",
            "What was the name of the first island that Guliver got to?",
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    )

    demo.launch(share=False)
