# -*- coding: utf-8 -*-
'''
将一张图片切为9张图
'''
import subprocess
import os
from PIL import Image

from PIL import Image
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import PIL.Image as im



# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
        new_image.paste(image, (0, int((new_image_length - height) / 2)))  # (x,y)二元组表示粘贴上图相对下图的起始位置
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image


# 切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 3):
        for j in range(0, 3):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]

    return image_list,item_width

# 保存图片
def save_images(image_list):
    index = 0
    for image in image_list:
        image.save('./yolov5-master/img/' + str(index) + '.png', 'PNG')
        index += 1

def det_health_tag(img_path, tag_path = 'r1.png'):
    MIN_MATCH_COUNT = 3
    # img1 = cv2.imread('test2.png', 1)
    img1 = cv2.imread(tag_path, 1)
    # img2 = cv2.imread('2.jpg',1)
    img2 = img_path
    # 使用SIFT检测角点
    sift = cv2.xfeatures2d.SIFT_create()
    # 获取关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    flag = 0
    # 定义FLANN匹配器
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 使用KNN算法匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 去除错误匹配
    good = []
    for m, n in matches:
        if m.distance <= 0.7 * n.distance:
            good.append(m)

    # 单应性
    if len(good) > MIN_MATCH_COUNT:
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ravel方法将数据降维处理，最后并转换成列表格式
        matchesMask = mask.ravel().tolist()
        # 获取img1的图像尺寸
        h, w, dim = img1.shape
        # pts是图像img1的四个顶点
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 计算变换后的四个顶点坐标位置
        dst = cv2.perspectiveTransform(pts, M)

        # 根据四个顶点坐标位置在img2图像画出变换后的边框
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        flag = 1
    else:
        # print("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        dst = []
        flag =0

    # 显示匹配结果
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    return np.int32(dst), flag

def detect_r(img):
    img = im.fromarray(np.uint8(img))
    #file_path = "2.jpg"
    #image = Image.open(file_path)
    # image.show()
    image = fill_image(img)
    new_img = np.array(image)
    image_list, item_width = cut_image(image)
    k = 0
    for i in image_list:
        img_tr = np.array(i)

        lines, flag = det_health_tag(img_tr)
        if flag == 1:

            lines[:,0,0] = (k%3) * item_width + lines[:,0,0]
            lines[:, 0, 1] = int(k/3) * item_width + lines[:,0,1]

            new_img = cv2.polylines(new_img, [np.int32(lines)], True, (0, 0, 255), 10, cv2.LINE_AA)
        k = k + 1
    return new_img

def det_r(img):
    img = im.fromarray(np.uint8(img))
    # file_path = "2.jpg"
    # image = Image.open(file_path)
    # image.show()
    image = fill_image(img)
    new_img = np.array(image)
    image_list, item_width = cut_image(image)
    save_images(image_list)
    res = subprocess.getoutput('/.local/lib/python3.8 ./yolov5-master/detect.py --weights ./yolov5-master/runs/train/exp4/weights/best.pt --source ./yolov5-master/img/ --device 0 --save-txt --exist-ok')

    label,vertices = tes()
    merge()
    #print(label,vertices)
    flag = len(label)
    return flag

def det_text():
    #res1 = subprocess.getoutput('conda activate yolov5')
    #res = subprocess.run("python3 ./yolov5-master/detect.py")
    cmd1 = "cd ./text"
    cmd2 = "python3 detect.py"
    cmd = cmd1 + "&&" + cmd2
    #res = subprocess.run(cmd, shell = True)



def tes():
    f1 = open('/home/trt/trt_pro/yolov5-master/detect/labels.txt')
    result = list()
    label=[]
    vertices = []
    for line in f1.readlines():  #
        vertice =list(map(float,line.rstrip('\n').lstrip('\ufeff').split(' ')[:5]))
        label.append(int(vertice[0]))
        vertices.append(vertice[1:5])
    return label,vertices

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 1280
    height_new = 720
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def merge():
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
    dir = '/home/trt/trt_pro/yolov5-master/detect'
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if "png" in file:
                all_path.append(os.path.join(root, file))
    #print(dirName)
    toImage = Image.new('RGBA', (width_i * line_max, height_i * row_max))

    for i in range(0, row_max):

        for j in range(0, line_max):
            pic_fole_head = Image.open(os.path.join(dir, str(i + j * 3) + '.png'))
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
    toImage = np.array(toImage)
    #toImage = cv2.cvtColor(toImage, cv2.COLOR_BGR2RGB)
    toImage = img_resize(toImage)
    cv2.imwrite(os.path.join('/home/trt/trt_pro', 'merged.png'), toImage)
    #toImage.save('merged.png')

if __name__ == '__main__':
    '''
    mg_cv1 = cv2.imread('2.jpg')
    img = detect_r(mg_cv1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    '''
    mg_cv1 = cv2.imread('2.jpg')
    img = im.fromarray(np.uint8(mg_cv1))
    # file_path = "2.jpg"
    # image = Image.open(file_path)
    # image.show()
    image = fill_image(img)
    new_img = np.array(image)
    image_list, item_width = cut_image(image)
    save_images(image_list)
    res = subprocess.getoutput('python ./yolov5-master/detect.py --weights ./yolov5-master/runs/train/exp4/weights/best.pt --source ./yolov5-master/img/ --device 0 --save-txt --exist-ok')
    #print(res)
    '''
    label,vertices = tes()
    print(label,vertices)
    print(len(label))
    for i in range(0,len(label)):
        line1 = np.int32(vertices[i])
        line1[0:2] = ((label[i]-1) % 3) * item_width + line1[0:2]
        line1[2:4] = int((label[i]-1) / 3) * item_width + line1[2:4]
        #new_img = cv2.polylines(new_img, [np.int32(vertices)], True, (0, 0, 255), 10, cv2.LINE_AA)
        print(line1)
    '''
    merge()
