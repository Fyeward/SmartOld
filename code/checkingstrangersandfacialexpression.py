# -*- coding: utf-8 -*-
'''
陌生人识别模型和表情识别模型的结合的主程序

用法：
python checkingstrangersandfacialexpression.py
python checkingstrangersandfacialexpression.py --filename tests/room_01.mp4
'''

# 导入包
import argparse
from oldcare.facial import FaceUtil
from PIL import Image, ImageDraw, ImageFont
from oldcare.utils import fileassistant
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import time
import numpy as np
import os
import imutils
import subprocess
import argparse
from flask import Flask, render_template, Response, request
from oldcare.camera import VideoCamera
import tensorflow as tf
graph = tf.get_default_graph()

# # 得到当前时间
# current_time = time.strftime('%Y-%m-%d %H:%M:%S',
#                              time.localtime(time.time()))
# # print('[INFO] %s 陌生人检测程序和表情检测程序启动了.'%(current_time))

# # 传入参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--filename", required=False, default = '',
# 	help="")
# args = vars(ap.parse_args())
# input_video = args['filename']

# 全局变量
global facial_recognition_model_path,facial_expression_model_path,output_stranger_path,output_smile_path,people_info_path,facial_expression_info_path,python_path
facial_recognition_model_path = '../models/face_recognition_hognew.pickle'
facial_expression_model_path = '../models/face_expressionCNNnew.hdf5'

output_stranger_path = '../supervision/strangers'
output_smile_path = '../supervision/smile'

people_info_path = '../info/people_info.csv'
facial_expression_info_path = '../info/facial_expression_info.csv'
# your python path
python_path = '/home/fyeward/anaconda3/envs/tf/bin/python3.6'

# 全局常量
global FACIAL_EXPRESSION_TARGET_WIDTH,FACIAL_EXPRESSION_TARGET_HEIGHT,VIDEO_WIDTH,VIDEO_HEIGHT,ANGLE
FACIAL_EXPRESSION_TARGET_WIDTH = 32
FACIAL_EXPRESSION_TARGET_HEIGHT = 32

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

ANGLE = 20

# 得到 ID->姓名的map 、 ID->职位类型的map、
#摄像头ID->摄像头名字的map、表情ID->表情名字的map
global id_card_to_name, id_card_to_type,facial_expression_id_to_name

id_card_to_name, id_card_to_type = fileassistant.get_people_info(
                                                people_info_path)
facial_expression_id_to_name=fileassistant.get_facial_expression_info(
                                          facial_expression_info_path)

# 控制陌生人检测
global strangers_timing
global strangers_limit_time
global strangers_start_time
strangers_timing = 0
strangers_start_time = 0 # 开始时间
strangers_limit_time = 2 # if >= 2 seconds, then he/she is a stranger.



# 控制微笑检测
global facial_expression_timing,facial_expression_start_time,facial_expression_limit_time,faceutil,facial_expression_model,count,vs,location
facial_expression_timing = 0 # 计时开始
facial_expression_start_time = 0 # 开始时间
facial_expression_limit_time = 2 # if >= 2 seconds, he/she is smiling
faceutil = FaceUtil(facial_recognition_model_path)
facial_expression_model = load_model(facial_expression_model_path)
# print('[INFO] 开始检测陌生人和表情...')
counter = 0
vs = cv2.VideoCapture(0)
location = 'room'


def CheckingStranger(frame1):
    global strangers_timing
    global strangers_limit_time
    global strangers_start_time
    global facial_expression_timing, facial_expression_start_time, facial_expression_limit_time, faceutil, facial_expression_model, count, vs, location
    global id_card_to_name, id_card_to_type, facial_expression_id_to_name
    global facial_recognition_model_path, facial_expression_model_path, output_stranger_path, output_smile_path, people_info_path, facial_expression_info_path, python_path

    frame = cv2.flip(frame1, 1)
    frame = imutils.resize(frame, width=VIDEO_WIDTH,
                           height=VIDEO_HEIGHT)  # 压缩，加快识别速度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

    face_location_list, names = faceutil.get_face_location_and_name(
        frame)

    # 得到画面的四分之一位置和四分之三位置，并垂直划线
    one_fourth_image_center = (int(VIDEO_WIDTH / 4),
                               int(VIDEO_HEIGHT / 4))
    three_fourth_image_center = (int(VIDEO_WIDTH / 4 * 3),
                                 int(VIDEO_HEIGHT / 4 * 3))

    cv2.line(frame, (one_fourth_image_center[0], 0),
             (one_fourth_image_center[0], VIDEO_HEIGHT),
             (0, 255, 255), 1)
    cv2.line(frame, (three_fourth_image_center[0], 0),
             (three_fourth_image_center[0], VIDEO_HEIGHT),
             (0, 255, 255), 1)

    # 处理每一张识别到的人脸
    for ((left, top, right, bottom), name) in zip(face_location_list,
                                                  names):

        # 将人脸框出来
        rectangle_color = (0, 0, 255)
        if id_card_to_type[name] == 'old_people':
            rectangle_color = (0, 0, 128)
        elif id_card_to_type[name] == 'employee':
            rectangle_color = (255, 0, 0)
        elif id_card_to_type[name] == 'volunteer':
            rectangle_color = (0, 255, 0)
        else:
            pass
        cv2.rectangle(frame, (left, top), (right, bottom),
                      rectangle_color, 2)

        # 陌生人检测逻辑
        if 'Unknown' in names:  # alert
            if strangers_timing == 0:  # just start timing
                strangers_timing = 1
                strangers_start_time = time.time()
            else:  # already started timing
                strangers_end_time = time.time()
                difference = strangers_end_time - strangers_start_time

                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))

                if difference < strangers_limit_time:
                    print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                else:  # strangers appear
                    event_desc = '陌生人出现!!!'
                    event_location = '房间'
                    print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))
                    cv2.imwrite(os.path.join(output_stranger_path,
                                             'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), frame)  # snapshot

                    # insert into database
                    command = '%s inserting.py --event_desc %s --event_type 2 --event_location %s' % (
                    python_path, event_desc, event_location)
                    p = subprocess.Popen(command, shell=True)

                    # 开始陌生人追踪
                    unknown_face_center = (int((right + left) / 2),
                                           int((top + bottom) / 2))

                    cv2.circle(frame, (unknown_face_center[0],
                                       unknown_face_center[1]), 4, (0, 255, 0), -1)

                    direction = ''
                    # face locates too left, servo need to turn right,
                    # so that face turn right as well
                    if unknown_face_center[0] < one_fourth_image_center[0]:
                        direction = 'right'
                    elif unknown_face_center[0] > three_fourth_image_center[0]:
                        direction = 'left'

                    # adjust to servo
                    if direction:
                        print('%d-摄像头需要 turn %s %d 度' % (counter,
                                                         direction, ANGLE))

        else:  # everything is ok
            strangers_timing = 0

        # 表情检测逻辑
        # 如果不是陌生人，且对象是老人
        if name != 'Unknown' and id_card_to_type[name] == 'old_people':
            # 表情检测逻辑
            roi = gray[top:bottom, left:right]
            roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                                   FACIAL_EXPRESSION_TARGET_HEIGHT))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # determine facial expression

            global graph
            with graph.as_default():
                (anger, neural, smile, surprised) = facial_expression_model.predict(roi)[0]
                print(neural, smile, anger, surprised)
                if (neural > smile) and (neural > anger) and (neural > surprised):
                    facial_expression_label = "Neural"
                elif (smile > neural) and (smile > anger) and (smile > surprised):
                    facial_expression_label = "Smile"
                elif (anger > neural) and (anger > smile) and (anger > surprised):
                    facial_expression_label = "Anger"
                else:
                    facial_expression_label = "Surprised"

                if facial_expression_label == 'Smile':  # alert
                    if facial_expression_timing == 0:  # just start timing
                        facial_expression_timing = 1
                        facial_expression_start_time = time.time()
                    else:  # already started timing
                        facial_expression_end_time = time.time()
                        difference = facial_expression_end_time - facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < facial_expression_limit_time:
                            print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            event_desc = '%s正在笑' % (id_card_to_name[name])
                            event_location = '房间'
                            print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))
                            cv2.imwrite(os.path.join(output_smile_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), frame)  # snapshot

                            # insert into database
                            command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                            python_path, event_desc, event_location, int(name))
                            p = subprocess.Popen(command, shell=True)

                elif facial_expression_label == 'Anger':  # alert
                    if facial_expression_timing == 0:  # just start timing
                        facial_expression_timing = 1
                        facial_expression_start_time = time.time()
                    else:  # already started timing
                        facial_expression_end_time = time.time()
                        difference = facial_expression_end_time - facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < facial_expression_limit_time:
                            print('[INFO] %s, 房间, %s仅怒了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            event_desc = '%s正在怒' % (id_card_to_name[name])
                            event_location = '房间'
                            print('[EVENT] %s, 房间, %s正在怒.' % (current_time, id_card_to_name[name]))
                            cv2.imwrite(os.path.join(output_smile_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), frame)  # snapshot

                            # insert into database
                            command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                            python_path, event_desc, event_location, int(name))
                            p = subprocess.Popen(command, shell=True)

                elif facial_expression_label == 'Surprised':  # alert
                    if facial_expression_timing == 0:  # just start timing
                        facial_expression_timing = 1
                        facial_expression_start_time = time.time()
                    else:  # already started timing
                        facial_expression_end_time = time.time()
                        difference = facial_expression_end_time - facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < facial_expression_limit_time:
                            print('[INFO] %s, 房间, %s仅惊了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            event_desc = '%s正在惊' % (id_card_to_name[name])
                            event_location = '房间'
                            print('[EVENT] %s, 房间, %s正在惊.' % (current_time, id_card_to_name[name]))
                            cv2.imwrite(os.path.join(output_smile_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        frame)  # snapshot

                            # insert into database
                            command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                                python_path, event_desc, event_location, int(name))
                            p = subprocess.Popen(command, shell=True)

                else:  # everything is ok
                    facial_expression_timing = 0
        else:  # 如果是陌生人，则不检测表情
            facial_expression_label = ''

        # 人脸识别和表情识别都结束后，把表情和人名写上
        # (同时处理中文显示问题)
        img_PIL = Image.fromarray(cv2.cvtColor(frame,
                                               cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img_PIL)
        final_label = id_card_to_name[name] + ': ' + facial_expression_id_to_name[
            facial_expression_label] if facial_expression_label else id_card_to_name[name]
        draw.text((left, top - 30), final_label,
                  font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
                  fill=(255, 0, 0))  # linux

        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

    # show our detected faces along with smiling/not smiling labels
    # cv2.imshow("Checking Strangers and Ole People's Face Expression",
    #            frame)
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

app = Flask(__name__)
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

    # if video_camera is None:
    #     video_camera = VideoCamera()

    while True:
        ret, frame = vs.read()
        frame = cv2.flip(frame, 1)
        frame = CheckingStranger(frame)

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.43.5', threaded=True, port=5001)