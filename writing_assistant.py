# ✅ Install required packages
# (already handled in requirements.txt)

import os
from google.colab import userdata
import gradio as gr
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool

# ✅ Load API key from Colab secrets
try:
    groq_api_key = userdata.get('GROQ_API_KEY')
except Exception as e:
    raise ValueError("⚠️ Failed to load Groq API Key from Colab secrets")

# ✅ Writing Knowledge Base
WRITING_KNOWLEDGE = {
    "hook": "A hook is a compelling first sentence designed to grab the reader's attention.",
    "thesis": "A thesis statement clearly expresses the main idea of your paper.",
    "conclusion": "A strong conclusion summarizes your argument and reinforces your thesis."
}

# ✅ Tool 1: Writing Knowledge Base
@tool
def writing_knowledge_base(topic: str) -> str:
    topic = topic.lower().strip()
    if topic in WRITING_KNOWLEDGE:
        return WRITING_KNOWLEDGE[topic]
    for key in WRITING_KNOWLEDGE:
        if key in topic:
            return WRITING_KNOWLEDGE[key]
    return ""

# ✅ Tool 2: Reading Time Calculator
@tool
def estimate_reading_time(words: str) -> str:
    try:
        word_count = int(''.join(filter(str.isdigit, words)))
        if word_count <= 0:
            return "Word count must be positive"
        minutes = word_count / 200
        if minutes < 1:
            return f"Exact reading time: {int(word_count / 200 * 60)} seconds ({word_count} words)"
        elif minutes > 60:
            hours = int(minutes // 60)
            remaining_mins = int(minutes % 60)
            return f"Exact reading time: {hours}h {remaining_mins}m ({word_count} words)"
        else:
            return f"Exact reading time: {minutes:.1f} minutes ({word_count} words)"
    except:
        return "Please enter like: 1200 or '1200 words'"

tools = [
    Tool(name="WritingGuide", func=writing_knowledge_base, description="For writing terms like hook, thesis, conclusion"),
    Tool(name="ReadingTime", func=estimate_reading_time, description="For calculating reading time from word count")
]

writing_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
You are a professional writing assistant. Provide short, clear, and helpful responses in this format:

Definition (1 sentence)
- Bullet point 1
- Bullet point 2

Only include an example if it's very helpful.

Examples:

Q: What is a thesis statement?
A: A thesis statement summarizes the main argument of your essay in one sentence.
- Guides the essay's structure and focus
- Should be clear, specific, and arguable

Q: What is a hook in writing?
A: A hook is the opening sentence that grabs the reader’s attention.
- Can be a question, quote, or bold statement
- Makes the reader want to keep reading

Q: {task}
"""
)

llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    groq_api_key=groq_api_key
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
    max_iterations=2,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

def writing_assistant(user_input):
    if not user_input.strip():
        return "❌ Please enter a question."

    kb_answer = writing_knowledge_base.invoke(user_input)
    if kb_answer and kb_answer.strip() != "":
        return f"🛠️ Used Knowledge Base:\n\n{kb_answer}"

    if any(word in user_input.lower() for word in ["word", "read", "minute", "hour", "'", "\""]):
        nums = []
        for word in user_input.split():
            clean_word = word.strip('\"\'').replace(',', '')
            if clean_word.isdigit():
                nums.append(int(clean_word))
        if nums:
            return f"🛠️ Used Reading Calculator:\n\n{estimate_reading_time.invoke(str(nums[0]))}"

    try:
        response = agent.invoke({
            "input": writing_prompt.format(task=user_input)
        })
        return f"🤖 Used LLM:\n\n{response['output']}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

gr.Interface(
    fn=writing_assistant,
    inputs=gr.Textbox(label="✍️ Enter your question", placeholder="e.g., What is a thesis statement?"),
    outputs=gr.Textbox(label="🧠 Assistant Response"),
    title="📝 AI Writing Assistant",
    description="Ask about writing techniques, terms, or reading time. Powered by LangChain, Groq & LLaMA 4."
).launch()