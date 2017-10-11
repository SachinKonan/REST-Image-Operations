from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask import request
from skimage import feature
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

mlp = joblib.load('/home/michaelnibble12121212/mysite' + '/finalized_model.sav')

app = Flask(__name__)
cors = CORS(app)

tasks = {"tasks": {'a': 1, 'b': 2, 'c': 3}}
string = ''


def liner(x0, y0, x1, y1, img):
    #uses brensenham's approach
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    ix = 1 if(x1 > x0) else -1
    iy = 1 if(y1 > y0) else -1
    if(dx < dy):
        ix,iy = iy, ix
        dx,dy = dy, dx
    p = 2*dy - dx
    y = y0
    for x in range(x0, x1 + ix, ix):
        img[y,x] = [255, 0, 0]
        if(p > 0):
             y += iy
             p = p - 2*dx
        p = p + 2*dy
    return img

def convertback(img):
        image = Image.fromarray(img)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue())
        return img_str.decode('utf-8')

@app.route('/')
def hello_world():
    return 'Hello from Sachin Flask!'

@app.route('/getval')
def getter():
    return '3'

@app.route('/get/<int:task_id>', methods=['GET'])
def get_task(task_id):
    tasks[str(task_id)] = task_id
    return jsonprinter(tasks)

@app.route('/cvt2gray', methods = ['POST'])
def getbw():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    def cvt2Gray(img):
        return np.dot( img[:,:,:3],[0.2989, 0.5870, 0.1140] )
    ret = convertback(cvt2Gray(img1))
    return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtgetRed', methods = ['POST'])
def getRed():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    ret = convertback(img1[:,:, 0])
    return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtgetBlue', methods = ['POST'])
def getBlue():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    ret = convertback(img1[:,:, 1])
    return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtgetGreen', methods = ['POST'])
def getGreen():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    ret = convertback(img1[:,:, 2])
    return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtinRange', methods = ['POST'])
def masker():
    data = request.get_json()
    string = data['imgdata'].split(',')
    lower = data['lower']
    upper = data['upper']
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    for i in range(0, img1.shape[0]):
        for x in range(0, img1.shape[1]):
            j = img1[i, x]
            if( (j[0] > lower[0] and j[0] < upper[0]) and (j[1] > lower[1] and j[0] < upper[1]) and (j[2] > lower[2] and j[2] < upper[2]) ):
                img1[i,x] = 1
            else:
                img1[i,x] = 0
    ret = convertback(img1)
    return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtCrop', methods = ['POST'])
def cropper():
    data = request.get_json()
    dim = data['data']
    string = data['imgdata'].split(',')
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    curr_shape = img1.shape
    tasks['shape'] = curr_shape
    if('height' not in data.keys()):
        height = curr_shape[0]
    else:
        height = data['height']
    if('width' not in data.keys()):
        width = curr_shape[1]
    else:
        width = data['width']

    if(height > curr_shape[0] or width > curr_shape(1)):
        return jsonify({'ERROR': 'WIDTH AND HEIGHT ARE WRONG DIMENSIONS'})
    else:
        ret = convertback(img1[:height, :width])
        return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtdrawRect', methods = ['POST'])
def drawRect():
    data = request.get_json()
    string = data['imgdata'].split(',')
    # format: upper left, upper right, lower right, lower left
    #x,y = col, row, must be switched
    tasks['stuff'] = data
    coordinates = data['data']
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    replace = [255,0,0] if(img1.shape[2] == 3) else [255,0,0,0]
    start = coordinates.get('px')
    width = coordinates.get('width')
    height = coordinates.get('height')

    if(any((start, width, height)) is None):
        return jsonify('Not all args specified')

    elif(start[1] > img1.shape[0] or start[0] > img1.shape[1]):
        return jsonify('Initial pixel is out of range')

    elif(start[1] + width > img1.shape[0] or start[0] + height > img1.shape[1]):
        return jsonify('Rect doesnt fit on this thing')

    else:
        img1[start[1]: start[1] + height, start[0]] = replace
        img1[start[1], start[0]: start[0] + width] = replace
        img1[start[1]: start[1] + height, start[0] + width] = replace
        img1[start[1] + height, start[0]:start[0] + width] = replace
        ret = convertback(img1)
        return jsonify("data:image/jpeg;base64," + ret)

@app.route('/cvtCannyEdge', methods = ['POST'])
def can():
	data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
	gray = cvt2Gray(img1)
    edged = feature.canny(gray).astype(float)*255
	ret = convertback(edged)
	return jsonify("data:image/jpeg;base64," + ret)

def list_flattener(l, current = []):
	for i in l:
		if type(i) is list:
			list_flattener(i, current)
		else:
			current.append(i)
	return current
			
@app.route('/cvtdrawLine', methods=['POST'])
def linedrawer():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
	#should be a list with two lists that show endpoints: [ [x1,y1], [x2,y2] ]
	coordinates = data['coordinates']
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
	new_img = liner(list_flattener(coordinates), img1)
	ret = convertback(new_img)
    return jsonify("data:image/jpeg;base64," + ret)

@app.route('/eyedetector', methods=['POST'])
def eyedetection():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    tasks['image'] = list(img1)
    i = img1

    predict = mlp.predict(i.reshape(1,9216))
    x_points = predict[0][::2]
    y_points = predict[0][1::2]

    plt.imshow(i.reshape(96,96), cmap='gray')
    plt.scatter(x_points * 48 + 48, y_points * 48 + 48, marker='x', s=50)
    buf = BytesIO()
    plt.savefig(buf, format='JPEG')

    def convertback(buffer):
        img_str = base64.b64encode(buffer)
        return img_str.decode('utf-8')

    returnval = convertback(buf)

    return jsonify("data:image/jpeg;base64," + returnval)
	
@app.route('/store', methods=['POST'])
def getdata():
    data = request.get_json()
    string = data['imgdata'].split(',')
    tasks['header'] = string[0]
    return jsonify("Done")

	
@app.route('/printer')
def printer():
     return string


def jsonprinter(d,indent = '', end = True):
    string = '{' + '<br/>'
    for k, v in d.items():
        if(not isinstance(v, dict)):
            string += indent + k + ':' + str(v) + ','
        else:
            string += indent + k + ':' + jsonprinter(v, indent = indent+ '    ', end = False)
        string += '<br/>'
    returner = (string + '%s' + '}') % ('    ' * (int(len(indent)/4) - 1))
    if end:
        return returner
    else:
        return returner + ','

print(2)

"""
var url = "https://sachk480.pythonanywhere.com/get"

$.ajax({
  url: url,
  data: {"tasks": 1, "image": 2},
  success: function(data){
  	console.log(data)
  },
  dataType: "json",
});
"""
# Simple Hello World APp