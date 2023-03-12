from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from core.complexity.complexity_core import get_complexity
from core.generalization.generalization_core import get_generalization
from core.pruning.pruning_core import get_pruning
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = 'upload'


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/complexity', methods=['POST'])
def complexity_measure():
    # 在这里添加计算模型复杂度的代码，返回一个包含8个指标值的JSON响应
    data = request.get_json()
    selectedTask = data.get('selectedTask')
    print(selectedTask)
    metrics = get_complexity(selectedTask)
    return jsonify(metrics)


@app.route('/generalization', methods=['POST'])
def generalization_measure():
    # 在这里添加计算模型泛化性度量的代码，返回一个包含2个值的JSON响应
    data = request.get_json()
    print(data.get('selectedTask'))
    metrics = get_generalization(data)
    return jsonify(metrics)


@app.route('/pruning', methods=['POST'])
def pruning():
    data = request.get_json()
    print(data.get('selectedTask'))
    metrics = get_pruning(data.get('selectedTask'))
    return jsonify(metrics)


@app.route('/upload', methods=['POST'])
def upload_file():
    # 获取上传的文件
    file = request.files['file']
    # 获取文件名
    filename = secure_filename(file.filename)
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core/upload')
    # 保存文件到UPLOAD_FOLDER目录中
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    # 返回上传成功的消息
    return jsonify({'msg': '上传成功'})


@app.route('/download', methods=['GET'])
def download():
    return send_file('./core/upload/finetuned_model.tar')


if __name__ == '__main__':
    app.run()
