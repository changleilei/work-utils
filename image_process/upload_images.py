# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/6/30 3:41 下午
==================================="""
import os
import re

import cv2
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

url_ = "http://192.168.0.181:9000"


class LabelCenter(object):
    @staticmethod
    def _get_cookie():
        with requests.session() as se:
            res = se.get(url_+'/user/login')
            csrf_token_list = re.findall('<input type="hidden" name="csrfmiddlewaretoken" value=(\W+.*)>', res.text)
            if csrf_token_list:
                csrf_token = csrf_token_list[0].replace("\"", '')
                res = se.post(url_+'/user/login/?next=/projects/', data={
                    'csrfmiddlewaretoken': csrf_token,
                    'email': '852324033@qq.com',
                    'password': '12345678'
                })
                return res.cookies.get('sessionid', '')

    def get_token(self):
        ck = self._get_cookie()
        res = requests.get(url_+'/api/current-user/token', cookies={'sessionid': ck})
        return res.json().get('token')

    def send_raw_img_to_label_center(self, frame, project_id=65):
        try:
            token = self.get_token()
        except:
            return
        t = cv2.imread(frame)
        _, encoded_image = cv2.imencode(".jpg", t)
        img_bytes = encoded_image.tobytes()

        multipart_encoder = MultipartEncoder(fields={
            "image_type": "message",
            "name": "image",
            "type": "image/jpg",
            "filename": "image.jpg",
            "Content-Type": "multipart/form-data",
            "image": (
                "image.jpg", img_bytes, "image/jpg")
        }
        )
        headers = {'Authorization': 'Token {}'.format(token), 'Content-Type': multipart_encoder.content_type}
        try:
            requests.post(url=url_+'/api/projects/{}/import'.format(project_id), headers=headers,
                          data=multipart_encoder, timeout=2)
        except:
            pass

    def delete_task(self, task_id=140001):
        ck = self._get_cookie()
        url = url_+'/api/tasks/{}/'.format(task_id)
        res = requests.delete(url, cookies={'sessionid': ck})
        return res.status_code


if __name__ == '__main__':
    c = LabelCenter()
    datas_path = os.listdir('/data/cll/work-utils/data/images')
    for d in datas_path:

        c.send_raw_img_to_label_center('/data/cll/work-utils/data/images' + os.sep + d)
