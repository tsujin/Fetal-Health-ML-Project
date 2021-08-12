import csv

from flask import Flask
from flask import render_template
from flask import request
import sys

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == "GET":
        return render_template('data.html')
    elif request.method == "POST":
        results = []

        user_csv = request.form.get('user_csv').split('\n')
        reader = csv.DictReader(user_csv)

        for row in reader:
            results.append(dict(row))

        print(results)

        return 'post'


if __name__ == '__main__':
    app.run()
