from flask import Flask
import numpy as np
import json
from flask import jsonify
from algo import calculate


app=Flask(__name__)

app.debug = True
app.config.update(DEBUG=True)

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

@app.route("/calc", methods=['POST'])
def calc():
    score = calculate()
    return jsonify(score)



if __name__ == '__main__':
    app.run()