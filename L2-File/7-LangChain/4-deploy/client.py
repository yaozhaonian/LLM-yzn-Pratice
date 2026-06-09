#使用LangChain编写客户端访问我们基于LangServer的WEB服务
#对于其他编程语言来说，可以使用RESTful API来调用我们的服务
#已经启动的service服务以及在新开的终端中运行client.py，返回:I enjoy programming.

from langserve import RemoteRunnable

if __name__ == "__main__":
    # 创建了一个chain对象()
    client = RemoteRunnable("http://localhost:8088/tslServer")
    print(client.invoke({'language': '英文', 'text': '我喜欢编程'}))