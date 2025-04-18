# 📝 AI Writing Assistant

An intelligent writing assistant built with LangChain, Groq, and Gradio.

## Features

- ✍️ Writing tips and definitions (hook, thesis, conclusion)
- ⏱️ Reading time estimator
- 🤖 LLM-based suggestions using LLaMA

## Run Locally

```bash
git clone https://github.com/your-username/ai-writing-assistant.git
cd ai-writing-assistant
pip install -r requirements.txt
python writing_assistant.py
```

## Run with Docker

```bash
docker build -t writing-assistant .
docker run -p 7860:7860 writing-assistant
```