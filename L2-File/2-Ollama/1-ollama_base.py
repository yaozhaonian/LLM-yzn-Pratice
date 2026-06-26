from ollama import Client

# 创建Ollama客户端实例
client = Client(host='http://127.0.0.1:11434')  # ollama默认端口11434
# 获取模型列表并打印端口和访问链接
# models = client.list()
# for model in models:
#     print(model)

print('=' * 50)

def get_embedding(text, model='bge-m3:latest'):
    # 使用ollama库获取嵌入向量
    response = client.embed(model, text)
    print("原始向量化响应", response)
    print('=' * 50)
    embedding = response['embeddings']
    return embedding

test_query = "我爱你"
# test_query = ["我爱你", 'hello我是Jeff，I am a goodman']

# vec = get_embedding(test_query)
# print(vec)
# #  "我爱你" 文本的嵌入表示的维度。
# print("维度:",len(vec))
# print('=' * 50)
# print("维度:",len(vec[0]))
# print('=' * 50)
def chat_with_model(messages, model='qwen2.5:7b'):
    response = client.chat(model=model, messages=messages)
    return response["message"].content
print("与大模型通话\n大模型回复:",chat_with_model(messages=[{"role": "user", "content": "你好,你是谁？"}]))