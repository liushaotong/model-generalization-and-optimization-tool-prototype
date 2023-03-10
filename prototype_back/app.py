from flask import Flask, request, jsonify
from flask_cors import CORS
from core.complexity.complexity_core import get_complexity
from core.generalization.generalization_core import get_generalization
from core.pruning.pruning_core import get_pruning
app = Flask(__name__)
cors = CORS(app)


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


if __name__ == '__main__':
    app.run()
