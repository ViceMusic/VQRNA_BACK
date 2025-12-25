# 根目录组件
from flask import Flask, request, jsonify
import json
from flask_cors import CORS  # Import CORS
from code.DB import SQLiteDB


def space():
    # 新建一个作用域，防止变量污染
    sqlitedb=SQLiteDB()
    sqlitedb.connect()


    motifs=sqlitedb.fetch_all_as_dicts("motif")
    print(motifs)
    motifs=sqlitedb.fetch_all_as_dicts("motif")
    print(motifs)


space()

exit("就测试用，不要运行这个文件")



def new_client(client):
    print("新的客户端连接, 暂时不支持多线程运行:", client['id'])


def expose_sentence(seq, message):
    return ""


def expose_fasta(fasta_name, message):
    return ""

# parameters：taskid，userid，state，
def generateResponse():
    temp={
        "test":1,
        "number":2
    }
    return json.dumps(temp) 


# Flask的部分

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/test', methods=['GET','POST'])
def test():
    return 'Hello World'


@app.route('/submit/seq', methods=['POST'])
def handle_data():
    # 判断 content-type
    content_type = request.headers.get('Content-Type')
    print(request.data)  # 目前状态是前端可以正常发过来消息，但是后端暂时还不知道该怎么回复
    if content_type == 'application/json':

        return generateResponse(), 200

    else:
        return 'Unsupported POST', 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=6077)
