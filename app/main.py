from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import torch
from torchvision import transforms
from net.bgnet import Net
import time
from PIL import Image
import numpy as np
from video_function import split_video_frame, run_model, process_seg, combine_frame

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './video'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'flv', 'wmv'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # max upload - 50mb

# Load the model
model = Net()
model.load_state_dict(torch.load('BGNet.pth'))
model.cuda()
model.eval()  # Set the model to evaluation mode

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('video_display', filename=filename))
    return render_template('upload.html')

def gen_frames(filename):
    # video info
    video_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # 영상 프레임 분할
    imgs = split_video_frame(video_file)

    # segmentation model
    segs = run_model(imgs, 'BGNet.pth')  # model pth 파일 경로 지정

    # segmentation 가공
    processed_segs, seg_ratios = process_seg(segs)

    # image + segmentation
    results = combine_frame(imgs, processed_segs, 0.4)

    for result in results:
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result = cv2.imencode('.jpg', result)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')
        time.sleep(0.025)  # wait for 0.1 second

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(gen_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_display/<filename>')
def video_display(filename):
    return render_template('video_feed.html', video_feed_url=url_for('video_feed', filename=filename))

if __name__ == '__main__':
    app.run()  # removed debug=True

