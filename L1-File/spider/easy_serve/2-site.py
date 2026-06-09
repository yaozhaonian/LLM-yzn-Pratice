# 基于flask框架搭建的网站
from flask import Flask, render_template, request, jsonify
import datetime as da
app = Flask(__name__, template_folder='html', static_folder='static')

@app.route('/')
def home():
    goods = ['good1', 'good2', 'good3'] # 传输数据
    return render_template('home.html',**{"goods":goods})

@app.get('/index')
def index():
    timer = da.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return render_template(template_name_or_list='index.html', **{"timer":timer})

app.run()
