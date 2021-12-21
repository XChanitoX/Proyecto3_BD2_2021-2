import json
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/result')
def Result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)
