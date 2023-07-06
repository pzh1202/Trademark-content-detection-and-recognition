import requests
import re
import os
import time
import paddlehub as hub
import cv2


ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
def reg(img_path1, name):
    flag = 0
    np_images = [cv2.imread(img_path1)]
    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.5)  # 识别中文文本置信度的阈值；

    #img_cv1 = cv2.cvtColor(img_cv1, cv2.COLOR_BGR2RGB)

    for result in results:
        data = result['data']
        save_path = result['save_path']

        for infomation in data:
            if name in infomation['text']:
                flag = 1

    return flag

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
name = input('您要爬取什么图片')
num = 0
num_1 = 0
num_2 = 0
x = input('您要爬取几张呢?，输入1等于60张图片。')
list_1 = []
for i in range(int(x)):
    name_1 = os.getcwd()
    name_2 = os.path.join(name_1,'name')
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+name+'&pn='+str(i*30)
    res = requests.get(url,headers=headers)
    htlm_1 = res.content.decode()
    a = re.findall('"objURL":"(.*?)",',htlm_1)
    if not os.path.exists(name_2):
        os.makedirs(name_2)
    for b in a:
        try:
            b_1 = re.findall('https:(.*?)&',b)
            b_2 = ''.join(b_1)
            if b_2 not in list_1:
                num = num +1
                img = requests.get(b)
                f = open(os.path.join(name_1,'图片',name+str(num)+'.jpg'),'ab')
                print('---------正在下载第'+str(num)+'张图片----------')
                f.write(img.content)
                f.close()
                list_1.append(b_2)
            elif b_2 in list_1:
                num_1 = num_1 + 1
                continue
        except Exception as e:
            print('---------第'+str(num)+'张图片无法下载----------')
            num_2 = num_2 +1
            continue
print('下载完成,总共下载{}张,成功下载:{}张,重复下载:{}张,下载失败:{}张'.format(num+num_1+num_2,num,num_1,num_2))
