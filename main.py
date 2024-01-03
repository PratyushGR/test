import os
import sys
import json

from google.colab.output import eval_js
from flask import Flask, render_template, request, redirect, url_for

sys.path.append(os.getcwd()+'/test/Dist')
from scripts import distreq
sys.path.append(os.getcwd())

print('Open URL: ',eval_js('google.colab.kernel.proxyPort(5000)'))
app = Flask(__name__, template_folder=os.getcwd()+'/test/Dist/templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    json_input = json.loads(request.form['json_input'])
    print(json_input.keys())
    return json_input

if __name__ == '__main__':
    app.run()
