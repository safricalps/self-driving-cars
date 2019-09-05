from flask import Flask
from flask_socketio import SocketIO, emit
import DQN
import json

model = DQN.DeepQNetwork(15,5,0.5)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def hello_world():
    return 'Hello World!'

# 这里是服务器监听的事件名
@socketio.on('updateJson')
def UpdateJsonHandler(data):

    data = dealFunc(data['data'])
    data = json.dumps(data.item())
    print(type(data))
    # update为前端定义的接受json的事件名
    emit("update", data)

def dealFunc(data):
    print(type(data))
    print(data)
    enviroment_data = []
    for i in range(5):
        enviroment_data.append(data['cars'][i]['x'])
        enviroment_data.append(data['cars'][i]['y'])
        enviroment_data.append(data['cars'][i]['speed'])
    status = data['status']
    if status == 1:
        return model.update(-1, enviroment_data, status)
    else:
        return model.update(0.1, enviroment_data, status)   

if __name__ == '__main__':
    socketio.run(app, "127.0.0.1", debug=True)
