import argparse
import functools
import os
import time
import uuid

import numpy as np
import torch
from flask import request, Flask, render_template
from flask_cors import CORS

from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
# 允许跨越访问
CORS(app)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('input_shape',      str,    '(1, 257, 257)',          '数据输入的形状')
add_arg('threshold',        float,   0.7,                     '判断是否为同一个人的阈值')
add_arg('audio_db',         str,    'audio_db',               '音频库的路径')
add_arg('model_path',       str,    'models/infer/model',     '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

device = torch.device("cuda")

# 加载模型
model = torch.jit.load(args.model_path)
model.to(device)
model.eval()

person_feature = []
person_name = []


def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32)
    # 执行预测
    feature = model(data)
    return feature.numpy()


# 加载要识别的音频库
def load_audio_db(path):
    name = os.path.basename(path)[:-4]
    feature = infer(path)
    person_name.append(name)
    person_feature.append(feature)
    print("Loaded %s audio." % name)


# 计算对角余弦值
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# 声纹识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    start_time1 = time.time()
    f = request.files['audio']
    if f:
        file_path = os.path.join('audio', str(uuid.uuid1()) + "." + f.filename.split('.')[-1])
        f.save(file_path)
        name = ''
        score = 0
        try:
            feature = infer(file_path)
            for i, person_f in enumerate(person_feature):
                # 计算相识度
                s = np.dot(feature, person_f.T)
                if s > score:
                    score = s
                    name = person_name[i]
            result = str({"code": 0, "msg": "success", "name": name}).replace("'", '"')
            print('duration:[%.0fms]' % ((time.time() - start_time1) * 1000), result)
            return result
        except:
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


# 声纹注册接口
@app.route("/register", methods=['POST'])
def register():
    global faces_db
    f = request.files['audio']
    user_name = request.values.get("name")
    if f or user_name:
        try:
            file_path = os.path.join(args.audio_db, user_name + "." + f.filename.split('.')[-1])
            f.save(file_path)
            load_audio_db(file_path)
            return str({"code": 0, "msg": "success"})
        except Exception as e:
            print(e)
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 2, "msg": "audio or name is None!"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    # 加载声纹库
    start = time.time()
    audios = os.listdir(args.audio_db)
    for audio in audios:
        path = os.path.join(args.audio_db, audio)
        load_audio_db(path)
    end = time.time()
    print('加载音频库完成，消耗时间：%fms' % (round((end - start) * 1000)))
    app.run(host='localhost', port=5000)
