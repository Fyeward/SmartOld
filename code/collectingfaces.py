# -*- coding: utf-8 -*-
'''
图像采集程序-人脸检测
由于外部程序需要调用它，所以不能使用相对路径

用法：
python collectingfaces.py --id 106 --imagedir /mnt/hgfs/code/OneTrueCode/images/faces

'''
import argparse
from oldcare.facial import FaceUtil
from oldcare.audio import audioplayer
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import shutil
import time
import argparse
from flask import Flask, render_template, Response, request
from oldcare.camera import VideoCamera

global audio_dir,error,start_time,limit_time,action_list,action_map,cam,faceutil,counter
global step, i
step = 0
i = 0
# 全局参数
audio_dir = '/mnt/hgfs/code/OneTrueCode/audios'

# 控制参数
error = 0
start_time = None
limit_time = 2  # 2 秒

# 传入参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-ic", "--id", required=True,
#                 help="")
# ap.add_argument("-id", "--imagedir", required=True,
#                 help="")
# args = vars(ap.parse_args())

action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
               'look_left', 'look_right', 'anger', 'surprised']
action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
              'smile': '请笑一笑', 'rise_head': '请抬头',
              'bow_head': '请低头', 'look_left': '请看左边',
              'look_right': '请看右边', 'anger': '请表现愤怒',
              'surprised': '请表现惊讶'}
# 设置摄像头
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

faceutil = FaceUtil()

counter = 0
def collectfaces():
    global audio_dir, error, start_time, limit_time, action_list, action_map, cam, faceutil, counter,step,image

    counter += 1
    _, image = cam.read()
    image = cv2.flip(image, 1)
    if counter <= 10:  # 放弃前10帧
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    if error == 1:
        end_time = time.time()
        difference = end_time - start_time
        print(difference)
        if difference >= limit_time:
            error = 0

    face_location_list = faceutil.get_face_location(image)
    for (left, top, right, bottom) in face_location_list:
        cv2.rectangle(image, (left, top), (right, bottom),
                      (0, 0, 255), 2)
    # cv2.imshow('Collecting Faces', image)  # show the image
    # Press 'ESC' for exiting video
    # k = cv2.waitKey(100) & 0xff
    # if k == 27:
    #     break

    face_count = len(face_location_list)
    if error == 0 and face_count == 0:  # 没有检测到人脸
        print('[WARNING] 没有检测到人脸')
        audioplayer.play_audio(os.path.join(audio_dir,
                                            'no_face_detected.mp3'))
        error = 1
        start_time = time.time()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    elif error == 0 and face_count == 1:  # 可以开始采集图像了
        print('[INFO] 可以开始采集图像了')
        audioplayer.play_audio(os.path.join(audio_dir,
                                            'start_image_capturing.mp3'))
        audioplayer.play_audio(os.path.join(audio_dir, action_list[step] + '.mp3'))
        step = 1
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    elif error == 0 and face_count > 1:  # 检测到多张人脸
        print('[WARNING] 检测到多张人脸')
        audioplayer.play_audio(os.path.join(audio_dir,
                                            'multi_faces_detected.mp3'))
        error = 1
        start_time = time.time()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    else:
        pass


global imagedir,id,action
counter = 0
imagedir = '../images/faces'
id = '404'
if os.path.exists(os.path.join(imagedir, id)):
    shutil.rmtree(os.path.join(imagedir, id), True)
os.mkdir(os.path.join(imagedir, id))

def collect():
    # 开始采集人脸
    global audio_dir, error, start_time, limit_time, action_list, action_map, cam, faceutil, counter,imagedir,id,action,image
    global i,step, action, counter
    action = action_list[step-1]

    action_name = action_map[action]
    print('%s-%d' % (action_name, i))
    _, img_OpenCV = cam.read()
    img_OpenCV = cv2.flip(img_OpenCV, 1)
    origin_img = img_OpenCV.copy()  # 保存时使用
    if i <= 4:
        i = i + 1
        ret, jpeg = cv2.imencode('.jpg', img_OpenCV)
        return jpeg.tobytes()
    if i>=20 :
        if step != 9:
            audioplayer.play_audio(os.path.join(audio_dir, action_list[step] + '.mp3'))
        else:
            print('[INFO] 采集完毕')
            audioplayer.play_audio(os.path.join(audio_dir, 'end_capturing.mp3'))
        step += 1
        i = 0
        counter = 0
        ret, jpeg = cv2.imencode('.jpg', img_OpenCV)
        return jpeg.tobytes()
    face_location_list = faceutil.get_face_location(img_OpenCV)
    for (left, top, right, bottom) in face_location_list:
        cv2.rectangle(img_OpenCV, (left, top),
                      (right, bottom), (0, 0, 255), 2)

    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV,
                                           cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_PIL)
    draw.text((int(image.shape[1] / 2), 30), action_name,
              font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
              fill=(255, 0, 0))  # linux

    # 转换回OpenCV格式
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),
                              cv2.COLOR_RGB2BGR)

    #cv2.imshow('Collecting Faces', img_OpenCV)  # show the image
    counter += 1
    i += 1
    image_name = os.path.join(imagedir, id,
                              action + '_' + str(counter) + '.jpg')
    cv2.imwrite(image_name, origin_img)
    ret, jpeg = cv2.imencode('.jpg', img_OpenCV)
    return jpeg.tobytes()



location = 'corridor'

# if location not in ['room', 'yard', 'corridor', 'desk']:
#     raise ValueError('location must be one of room, yard, corridor or desk')

# API
app = Flask(__name__)

video_camera = None
global_frame = None


@app.route('/')
def index():
    return render_template(location + '_camera.html')


@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera
    if video_camera == None:
        video_camera = VideoCamera()

    status = request.form.get('status')
    save_video_path = request.form.get('save_video_path')

    if status == "true":
        video_camera.start_record(save_video_path)
        return 'start record'
    else:
        video_camera.stop_record()
        return 'stop record'


def video_stream():
    global video_camera
    global global_frame
    global error,step
    # if video_camera is None:
    #     video_camera = VideoCamera()
    while step == 0 :
        frame = collectfaces()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


    while step <= 9:
        frame = collect()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/getMsg', methods=['GET', 'POST'])
def home():
    global id
    id = request.args["id"]
    name = request.args["name"]
    type = request.args["job"]
    if type=="老人":
        type = "old_people"
    print("id="+id)
    print("name="+name)
    print("job="+type)

    response = {
        'msg': 'Hello, Python !',
        'code': '200'
    }
    return response

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5004)
