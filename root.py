# 根目录组件
from flask import Flask, request, jsonify
import json
from flask_cors import CORS  # Import CORS
from MyCode.DB import SQLiteDB
from MyCode.Predict import predict_item

# 创建数据库
dbcontroller=SQLiteDB()

# 创建表result的检查
dbcontroller.create_results_table()

# 检查所有表的结构
tables = dbcontroller.list_tables()
print(tables)

# 检查result的结构
desc = dbcontroller.describe_table("results")
for col in desc:
    print(col)


# 测试阶段，每次都清理一下
# 这次先不清理了 dbcontroller.clear_results_table()
'''
插入任务记录或者返回对应的值
taskID = "task_001"    userID = "user_123"   timestamp = str(int(time.time()))
results = ["AUGCUA", None, "GGGAAA"]

record = dbcontroller.insert_task(
    taskID=taskID,
    userID=userID,
    timestamp=timestamp,
    results=results(所有结果的内容)
)

print(record), 插入以后就会返回对应的值

# 查询某个用户的所有任务
tasks = dbcontroller.get_tasks_by_userid("user_123")
print(tasks)
for t in tasks:
    print(t)

tasks=dbcontroller.get_all_tasks()
print(tasks)

exit('昨天晚上已经成功调试了内容，明天检查读取情况，以及异常数据的处理，先把所有内容读取到页面上')
'''




def space():
    # 新建一个作用域，防止变量污染
    sqlitedb=SQLiteDB()
    sqlitedb.connect()


    motifs=sqlitedb.fetch_all_as_dicts("motif")
    print(motifs)
    motifs=sqlitedb.fetch_all_as_dicts("motif")
    print(motifs)



def new_client(client):
    print("新的客户端连接, 暂时不支持多线程运行:", client['id'])


def expose_sentence(seq, message):
    return ""


def expose_fasta(fasta_name, message):
    return ""

# parameters：taskid，userid，state，
def generateResponse(data):
    print(data)
    # 需要塞进数据库的数据为：
    user_id=data["userid"]
    task_id=data["payload"]["task_id"]
    time_stamp=data["payload"]["timestamp"]

    # 准备映射后的结果
    seqs=data["payload"]["InformationBody"]
    results = []

    for seq in seqs:
        try:
            results.append(predict_item(seq))
        except Exception:
            print("!!! 捕捉到内部报错 !!!")
            traceback.print_exc() # 这行会打印出到底是哪个文件的哪一行报的 one_hot 错误
            results.append(None)

    # 将预测结果插入数据库中
    record = dbcontroller.insert_task(
        taskID=task_id,
        userID=user_id,
        timestamp=time_stamp,
        results=results
    )

    # print(record), 插入以后就会返回对应的值

    # res=predict_item(data["payload"]["InformationBody"])
    temp={
        "state":"success" if len(results)!=0 else "fail",
        "results":results
    }
    return json.dumps(temp) 


# Flask的部分

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# 用来测试的方法，对外暴露，检查是否崩了
@app.route('/test', methods=['GET','POST'])
def test():
    return 'Hello World'

# 单一提交的方法
@app.route('/query/tasks', methods=['POST',"GET","OPTIONS"])
def handle_tasks_by_userid():
    # 加上一个预先检测
    if request.method == "OPTIONS":
        return "", 204   # 直接放行，不碰业务逻辑
    # 判断 content-type
    content_type = request.headers.get('Content-Type')
    raw = request.data
    data = json.loads(raw.decode("utf-8"))
    userid=data["userid"]
    print(data)  # 目前状态是前端可以正常发过来消息，但是后端暂时还不知道该怎么回复
    tasks =[]
    try:
        tasks = dbcontroller.get_tasks_by_userid(userid)
    except Exception as e:
        print("出错了：", e)
    if content_type == 'application/json':
        return json.dumps({"userid":userid,"tasks":tasks}), 200
    else:
        return 'Unsupported POST', 400


# 单一提交的方法
@app.route('/submit/seq', methods=['POST'])
def handle_data():
    # 判断 content-type
    if request.method == "OPTIONS":
        return "", 204   # 直接放行，不碰业务逻辑
    content_type = request.headers.get('Content-Type')
    raw = request.data
    data = json.loads(raw.decode("utf-8"))
    print(data)  # 目前状态是前端可以正常发过来消息，但是后端暂时还不知道该怎么回复
    if content_type == 'application/json':

        return generateResponse(data), 200

    else:
        return 'Unsupported POST', 400

# Fasta文件提交的方法
@app.route('/submit/fasta', methods=['POST'])
def handle_data_fasta():
    if request.method == "OPTIONS":
        return "", 204   # 直接放行，不碰业务逻辑
    # 判断 content-type
    content_type = request.headers.get('Content-Type')
    raw = request.data
    data = json.loads(raw.decode("utf-8"))
    print(data)  # 目前状态是前端可以正常发过来消息，但是后端暂时还不知道该怎么回复
    if content_type == 'application/json':

        return generateResponse(data), 200

    else:
        return 'Unsupported POST', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=14444)
