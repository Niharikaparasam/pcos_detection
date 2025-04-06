from flask import Flask, send_from_directory
from flask_cors import CORS  # <--- import this
import os

app = Flask(__name__, static_folder='static')
CORS(app)  


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
