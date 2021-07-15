# -*- coding: utf-8 -*-
'''
将事件插入数据库主程序

用法：

'''

import datetime
import argparse
import json
from corpwechatbot.app import AppMsgSender
import requests as requests

f = open('allowinsertdatabase.txt','r')
content = f.read()
f.close()
allow = content[11:12]

if allow == '1': # 如果允许插入
    
    f = open('allowinsertdatabase.txt','w')
    f.write('is_allowed=0')
    f.close()
    
    print('准备插入数据库')
    
    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-ed", "--event_desc", required=False, 
                    default = '', help="")
    ap.add_argument("-et", "--event_type", required=False, 
                    default = '', help="")
    ap.add_argument("-el", "--event_location", required=False, 
                    default = '', help="")
    ap.add_argument("-epi", "--old_people_id", required=False, 
                    default = '', help="")
    args = vars(ap.parse_args())
    
    event_desc = args['event_desc']
    event_type = int(args['event_type']) if args['event_type'] else None
    event_location = args['event_location']
    old_people_id = int(args['old_people_id']) if args['old_people_id'] else None
    
    event_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    payload = {
               'event_desc':event_desc,
               'event_type':event_type,
               'event_location':event_location,
               'elder_name':old_people_id}
    
    print(payload)
    app = AppMsgSender(corpid='wwf8ccb05feff6e329',  # 你的企业id
                       corpsecret='nbmGEeBYIbuuSMNGNakovnjhq0-QkkG7PHdjMVVX_4w',  # 你的应用凭证密钥
                       agentid='1000002')  # 你的应用id
    app.send_text(content=payload["event_location"] + payload["event_desc"])
    r = requests.post("http://47.102.213.152:8080/EventManage/addEvent", data=json.dumps(payload),headers={'content-type': "application/json"})
    print(r.text)
    print(r.status_code)
    print('插入成功')

    f = open('allowinsertdatabase.txt','w')
    f.write('is_allowed=1')
    f.close()
else:
    print('just pass')

