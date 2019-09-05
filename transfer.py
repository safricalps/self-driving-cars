from flask import Flask
from flask_socketio import SocketIO, emit
import json
import DQN

model = DQN.DeepQNetwork(10,5,0.5)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def hello_world():
    return 'Hello World!'

# 这里是服务器监听的事件名
@socketio.on('UpdateJson')
def UpdateJsonHandler(data):

    data = json.loads(data)

    data = dealFunc(data)
    
    data = json.dumps(data)
    # update为前端定义的接受json的事件名
    emit("update", data)

def dealFunc(data):    
    enviroment_data = []
    for i in range(5):
        enviroment_data.append(data['cars'][i]['x'])
        enviroment_data.append(data['cars'][i]['y'])
        enviroment_data.append(data['cars'][i]['speed'])
    status = (enviroment_data['status'])
    if status == 1:
        next_action = model.update(-1, enviroment_data, status)
    else:
        next_action = model.update(0.1, enviroment_data, status)   
    
    return next_action

if __name__ == '__main__':
    socketio.run(app, "127.0.0.1", debug=True)
