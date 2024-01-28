import os
import openai
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from helper_function.helper import predict_tweet

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY") #replace with your openapi key like this " openai.api_key = '####' "

recent_post = []  # list to store the users inputted tweet/text
# this is to set the context of the chatgpt language model
messages = [
    {
        "role": "system",
        "content": f"""
Follow the steps strictly


STEP 1: Introduce yourself as Social media mental health specialist and your name is Lizzy.

Step 2: Tell the user in a polite manner to produce 2-10 of their most recent tweets or textual post on their various social media platform

Step 3: Put all the post in the python list {recent_post}

Step 4: Execute this function {predict_tweet(recent_post)} and output the function value in a descriptive and nice way.

""",
    }
]


def chatbot(user_inp):
    # user_inp is the users text
    if user_inp:
        # Append the user's text to the context set for the chatgpt language model
        messages.append({"role": "user", "content": user_inp})
        # Send the whole message list to the chatgpt ai using the ChatCompletion function
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        # put the response gotten from the chatgpt function to the reply variable
        reply = chat.choices[0].message.content
        # append the response of the chatgpt to the message list to store the conversation
        messages.append({"role": "assistant", "content": reply})
        return reply


# Interface of the chatbot built with gradio

inputs = gr.Textbox(lines=10, label="Chat with Lizzy")
outputs = gr.Textbox(label="Reply")

gr.Interface(
    fn=chatbot,
    inputs=inputs,
    outputs=outputs,
    title="Social Media Depression Chatbot Detector",
    description="Be Comfortable and Chat with lizzy",
    theme="compact",
).launch(share=False, debug=True)
