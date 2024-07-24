# Steps 

# SETUP
## 1. Install python dependencies

```
# install virtualenv package
python3 install virtualenv

# create the virtual environment
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the project dependencies
pip install -r requirements.txt
```

## 2. Add the Redis queue

```
docker run --name my-redis -p 6380:6379 -d redis/redis-stack:latest
```


## 3. Provide the OpenAI key
Speak with either Giani Statie or Dan Costin about how to procure a OpenAI API key that will be used to make requests to GPT.

That key should be then stored in a new `.env` file in this folder under the name `OPENAI_API_KEY=`

## (Outdated) Dowload llama-cpp 
* https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/blob/main/llama-2-13b-chat.Q5_K_M.gguf
* https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
* https://huggingface.co/TheBloke/firefly-llama2-7B-chat-GGUF/blob/main/firefly-llama2-7b-chat.Q5_K_M.gguf