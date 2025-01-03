import gradio as gr
import time

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def print_like_dislike(x: gr.LikeData, history: list):
    index = x.index[-1]
    if x.liked:
        print("The user liked the interaction")
    else:
        print("The user disliked the interaction")
    print("User request", history[index - 1])
    print("AI response", history[index])


def add_message(history, message):
    if message is not None:
        history.append({"role": "user", "content": message})
    return history, gr.Textbox(value=None, interactive=False)


def bot(history: list):
    response = "**That's cool!**"
    history.append({"role": "assistant", "content": ""})
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Welcome to BookWorm! 📚🐛
    Ask me about Machine Learning and I'll do my best to answer.
    """
    )

    chatbot = gr.Chatbot(label="", elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.Textbox(
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
        submit_btn=True,
    )

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.Textbox(interactive=True), None, [chat_input])

    like_btn = chatbot.like(print_like_dislike, chatbot, None, like_user_message=False, trigger_mode="once")

    gr.Markdown(
        """
    **Hint:** Use 👎 or 👍 on my answers to help me improve. 
    """
    )

demo.launch()
