import socket

s = socket.socket()

s.bind(('127.0.0.1', 8099))
s.listen(5)

while 1:
    # 等待客户端链接
    conn,addr = s.accept()

    # conn.recv()#接收数据
    # conn.send(b'hello')#发送数据

    data = conn.recv(1024)
    print(data.decode())
    with open('index.html') as f:
        data = f.read()
    res = "HTTP/1.1 200 ok\r\ncontent-type:text/html\r\n\r\n"
    res += data
    conn.send(res.encode())
    # res = "HTTP/1.1 200 ok\r\ncontent-type:text/html\r\n\r\n<h1>HEllo web,i am server.from python</h1><h2>this is h2 title</h2>".encode()
    # res = "HTTP/1.1 200 ok\r\ncontent-type:text/html\r\n\r\n".encode()
    # conn.send(res)
    # 在浏览器刷新是建立连接后浏览器向服务器发送了http协议包
    conn.close()
