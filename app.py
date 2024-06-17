from flask import Flask,request,jsonify
import numpy as np
import json
from flask import jsonify
from utils import *
from algo import calculate
import os
import shutil
import pandas as pd

app=Flask(__name__)

app.debug = True
app.config.update(DEBUG=True)

UPLOAD_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

## 设置json编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16,np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
app.json_encoder = NumpyEncoder

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/calc", methods=['POST'])
def calc():
    reqParams = json.loads(request.get_data())

    id = reqParams['id']
    clinical_data = reqParams['clinical_data']
    CT_list = reqParams['CT_data']['CT_list']

    score = calculate(id,clinical_data,CT_list)

    return jsonify({'score':score})


@app.route("/api/upload_CT", methods=['POST'])
def upload_CT():

    file = request.files.get('file')
    append_data = request.form.to_dict()
    name = append_data['name']
    id = append_data['nanoid']

    if not os.path.exists(f'data/CT/{id}'):
        os.makedirs(f'data/CT/{id}')
    if not os.path.exists(f'static/CT/{id}'):
        os.makedirs(f'static/CT/{id}')


    dcm_path = f"data/CT/{id}/{name}.dcm"
    jpg_path = f"static/CT/{id}/{name}.jpg"

    ## save dcm
    file.save(dcm_path)
    ## transfer dcm as jpg
    dcm_to_jpg(dcm_path,jpg_path)
        


    jpg_url = f"http://localhost:5000/{jpg_path}"


    return jsonify({'url':jpg_url,'name':name}),200


@app.route("/api/autofill_CSV", methods=['POST'])
def autofill_CSV():

    file = request.files.get('file')
    append_data = request.form.to_dict()
    id = append_data['nanoid']

    path=f"data/temp/{id}-autofill.csv"
    file.save(path)

    df = pd.read_csv(path)
    dict_data = df.to_dict(orient='list')
    for key in dict_data:
        dict_data[key] = dict_data[key][0]

    ##delete
    os.remove(path)

    return jsonify(dict_data),200



@app.route("/api/clear", methods=['POST'])
def clear():
    data = json.loads(request.get_data(as_text=True))
    id = data['id']

    dcm_path = f"data/CT/{id}"
    jpg_path = f"static/CT/{id}"
    if os.path.exists(dcm_path):
        shutil.rmtree(dcm_path)
    if os.path.exists(jpg_path):
        shutil.rmtree(jpg_path)

    return 'ok'

if __name__ == '__main__':
    app.run(debug=True)