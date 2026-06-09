# from bs4 import BeautifulSoup
# import requests

# url = 'https://www.sse.com.cn/disclosure/listedinfo/regular/'
# response = requests.get(url)

# response.encoding = 'utf-8'

# if response.status_code == 200:
#     soup = BeautifulSoup(response.text,'lxml')
#     title_tag = soup.find('title')

#     if title_tag:
#         print(title_tag.get_text())
#     else:
#         print('未找到标题')
# else:
#     print('请求失败,错误状态码:',response.status_code)


from bs4 import BeautifulSoup
import requests
# 指定网站，这个是巨潮资讯网
# url = 'https://www.cninfo.com.cn/new/index.jsp'
url = input('请输入网址：')

# 发送HTTP请求获取网页内容
response = requests.get(url)
# 中文乱码问题
response.encoding = 'utf-8'

soup = BeautifulSoup(response.text, 'lxml')

# 查找第一个 <a> 标签
# first_link = soup.find('a')
# print(first_link)
# print('-'*50)

# 获取第一个 <a> 标签的 href 属性
# first_link_url = first_link.get('href')
# print(first_link_url)
# print('-'*50)

# 查找所有 <a> 标签
# all_links = soup.find_all('a')
# print(all_links)

# all_text = soup.get_text()
# print('获取页面中所有文本内容\n',all_text)


# 查找第一个 <a> 标签的父标签
"""
first_link = soup.find('a')
parent_tag = first_link.parent
# children_tag = parent_tag.children
print('第一个 <a> 标签:',first_link)
print('='*30)
print('父标签:',parent_tag)
# print('='*30)
# print('父标签的子标签',children_tag)
"""

# 查找具有特定属性的标签
# table_find = soup.find_all('span',class_ = 'ell')
# print('查找\n',table_find)

# 查找具有 id="unique-id" 的 <input> 标签
# unique_input = soup.find('input', id='su')

# input_value = unique_input['value'] # 获取 input 输入框的值

# print(input_value)


# example_divs = soup.select('div.right-bar')
# print('example_divs的right_bar\n',example_divs)

# links = soup.select('a[href]')
# print('标签a的href属性\n',links)

# 查找嵌套的 <div> 标签
# https://www.cninfo.com.cn/new/index.jsp
print('看输出')
clearfix_divs = soup.find_all('div',class_ = 'clearfix')
for div in clearfix_divs:
    print(div.get_text(),'\n','-'*50,'\n')



