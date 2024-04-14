import gradio as gr
import os
from datetime import datetime
from multiprocessing import Process, Queue
from difflib import unified_diff
from IPython.display import display, HTML
import openai

from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper, AsyncHtmlLoader, Html2TextTransformer
import tiktoken

# 设置OpenAI API密钥
def set_openai_api_key(api_key: str):
    if api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key

# 使用Google搜索API获取搜索结果
def google_search(query: str, top_k: int = 1):
    search = GoogleSearchAPIWrapper(k=top_k)
    return search.results(query, top_k)


import openai

def local_search(query: str, top_k: int = 1):
    # 假设你有一个函数get_search_results，它可以从本地数据库中获取搜索结果
    search_results = get_search_results(query)

    # 如果找到了结果，返回前top_k个结果
    if search_results:
        return search_results[:top_k]
    else:
        # 如果没有找到结果，使用OpenAI的API来获取搜索结果
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            prompt=f"Search for '{query}' in the local database and list the steps.",
            temperature=0.5,
            max_tokens=100,
            n=1,  # 请求返回的完成数量
            stop=["\n\n"]  # 使用两个换行符作为完成的分隔符
        )
        # 解析API返回的文本，将其拆分成多个步骤
        steps_text = openai_response.choices[0].text.strip()
        steps = steps_text.split("\n")
        # 过滤空步骤并返回
        return [step for step in steps if step]

# 假设的get_search_results函数，仅作为示例
def get_search_results(query: str):
    # 这里应该是查询本地数据库的逻辑
    # 返回一个假设的搜索结果列表
    return []
def query_vector_database(step_content: str):
    # 这里应该是查询本地向量数据库的逻辑
    # 返回查询到的结果
    # 这里只是一个示例，实际实现可能会有所不同
    return "查询到的结果示例"

def analyze_with_model(step_content: str, query_result: str):
    # 使用OpenAI的API来分析步骤内容和查询结果
    prompt = f"Analyze the following step: '{step_content}' with its query result: '{query_result}'. Provide insights."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].text

def executeSteps(steps):
    model_responses = []
    for step in steps:
        # 对每个步骤执行本地向量库查询
        query_result = query_vector_database(step)
        # 检查query_result的token数量，如果太长，则进行切割
        query_result_chunks = chunk_text_by_sentence(query_result)
        # 初始化一个空字符串用于收集分析结果
        step_analysis = ""
        for chunk in query_result_chunks:
            # 对每个切割后的部分使用大模型进行分析
            chunk_analysis = analyze_with_model(step, chunk)
            # 将每个部分的分析结果拼接起来
            step_analysis += chunk_analysis + " "
        # 保存拼接后的大模型返回结果
        model_responses.append(step_analysis.strip())
    return model_responses




# 从URL获取页面内容
def fetch_page_content(url: str):
    loader = AsyncHtmlLoader([url])
    docs = loader.load()
    transformer = Html2TextTransformer()
    return transformer.transform_documents(docs)[0].page_content if docs else None

# 计算字符串的token数量
def count_tokens(text: str, encoding: str = "cl100k_base"):
    encoder = tiktoken.get_encoding(encoding)
    return len(encoder.encode(text))

# 按句子分块文本
def chunk_text_by_sentence(text: str, max_tokens: int = 2048):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if count_tokens('. '.join(current_chunk + [sentence])) <= max_tokens:
            current_chunk.append(sentence)
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    return chunks

# 获取草稿回答
def generate_draft_answer(question: str):
    chatgpt_prompt = f"You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: 2023-04. Current date: {datetime.now().strftime('%Y-%m-%d')}"
    draft_prompt = "Try to answer this question with step-by-step thoughts and make the answer more structural."
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", prompt=f"{chatgpt_prompt}\n\n{question}\n\n{draft_prompt}", temperature=1.0)
    return response.choices[0].text

# 显示文本差异
def display_text_diff(original: str, revised: str):
    diff = unified_diff(original.splitlines(keepends=True), revised.splitlines(keepends=True), fromfile='Original', tofile='Revised')
    diff_html = "".join(f"<div style='color:{'green' if line.startswith('+') else 'red' if line.startswith('-') else 'blue' if line.startswith('@') else 'black'};'>{line.rstrip()}</div>" for line in diff)
    display(HTML(diff_html))

# 主函数：处理问题并生成回答
def process_question(question: str):
    draft = generate_draft_answer(question)
    # 这里可以添加更多的处理步骤，例如分割草稿、查询、内容修订等
    final_answer = draft  # 假设最终答案就是草稿答案
    return draft, final_answer

# Gradio界面设置
def setup_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## RAT: Retrieval Augmented Thoughts Demo")
        question_input = gr.Textbox(label="Question", placeholder="Enter your question here")
        draft_output = gr.Textbox(label="Draft Answer")
        final_output = gr.Textbox(label="Final Answer")
        submit_btn = gr.Button("Submit")
        submit_btn.click(process_question,it_btn.click(process_question,
             inputs=[question_input],
             outputs=[draft_output, final_output])

        gr.Markdown("### Instructions")
        gr.Markdown("Enter your question in the textbox and click submit to see the draft and final answers.")

    return demo

# 启动Gradio界面
if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.launch(server_name="0.0.0.0", debug=True)