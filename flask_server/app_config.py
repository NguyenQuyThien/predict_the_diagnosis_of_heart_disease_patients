from flask import Flask
from datetime import timedelta
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-hahaha'
app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=10)
CORS(app)

