import os
from PIL import Image
import cv2

# import pyautogui
# import re
'''
把当前目录下的10*10张jpeg格式图片拼接成一张大图片

'''

# 图片压缩后的大小
width_i = 610
height_i = 610

# 每行每列显示图片数量
line_max = 3
row_max = 3

# 参数初始化
all_path = []
num = 0
pic_max = line_max * row_max

dirName = os.getcwd()
dir = '/home/peng/Downloads/pytorch_AdvancedEast-master/yolov5-master/detect'
for root, dirs, files in os.walk(dirName):
    for file in files:
        if "png" in file:
            all_path.append(os.path.join(root, file))
#print(dirName)
toImage = Image.new('RGBA', (width_i * line_max, height_i * row_max))

for i in range(0, row_max):

    for j in range(0, line_max):
        pic_fole_head = Image.open(os.path.join(dir, str(i+j*3) + '.png'))
        width, height = pic_fole_head.size

        tmppic = pic_fole_head.resize((width_i, height_i))

        loc = (int(i % line_max * width_i), int(j % line_max * height_i))

        # print("第" + str(num) + "存放位置" + str(loc))
        toImage.paste(tmppic, loc)
        num = num + 1
        if num >= len(all_path):
            #print("breadk")
            break

    if num >= pic_max:
        break

#print(toImage.size)
toImage = cv2.cvtColor(toImage, cv2.COLOR_BGR2RGB)
toImage = img_resize(toImage)
toImage.save('merged.png')