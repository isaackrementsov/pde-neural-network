from flask import Flask, render_template, request
from flask_socketio import SocketIO

import mysql.connector
import threading
import json

from neural_net import main


app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

credentials = json.load(open('./web_tracker/credentials.json', 'r'))
database = mysql.connector.connect(
    host=credentials['host'],
    user=credentials['user'],
    passwd=credentials['password'],
    database=credentials['database'],
    port=credentials['port']
)


@app.route('/', methods=['GET'])
def index():
    cursor = database.cursor()

    select_statement = 'SELECT * FROM pdenet.epochs;'
    cursor.execute(select_statement)
    epochs = cursor.fetchall()
    
    cursor.close()

    print(epochs)

    return render_template('index.html', epochs=epochs)


def initialize_network():
    main.run(socketio, database)


def start_sockets():
    parallel_thread = socketio.start_background_task(initialize_network)


def run():
    start_sockets()
    socketio.run(app)
