import argparse

import torch
from torchvision import transforms
from model import EAST
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw
from cut_img import detect_r,det_r

import cfg

from preprocess import resize_image
from preprocess_2 import getBinary
from preprocess_2 import verProject
from preprocess_2 import horProject
from preprocess_2 import removeContours
from preprocess_2 import roi
from nms import nms

import matplotlib.pyplot as plt

import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse

import os

import paddlehub as hub

from siamesemole.siamese import Siamese

import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import pyzbar.pyzbar as pyzbar
import numpy
from PIL import Image, ImageDraw, ImageFont
import cv2
import joblib
from skimage.feature import hog

import math



def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def load_pil(img):
    '''convert PIL Image to torch.Tensor
    '''
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    return t(img).unsqueeze(0)



def detect(img_path,image,image1, model,siamodel, device,pixel_threshold, save_path, numb, quiet=True):
    img = Image.open(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    with torch.no_grad():
        east_detect=model(load_pil(img).to(device))
    y = np.squeeze(east_detect.cpu().numpy(), axis=0)
    y[:3, :, :] = sigmoid(y[:3, :, :])

    print(y.shape)

    cond = np.greater_equal(y[0, :, :], pixel_threshold)
    #plt.imshow(img)

    plt.subplot(2, 2, 1)
    plt.imshow(np.array(y[0, :, :]))
    plt.subplot(2, 2, 2)
    plt.imshow(np.array(y[1, :, :]))
    plt.subplot(2, 2, 3)
    plt.imshow(np.array(y[2, :, :]))
    #plt.subplot(2, 2, 4)
    #plt.imshow(np.array(y[3, :, :]))
    plt.subplot(2, 2, 4)
    plt.imshow(img)
    plt.show()
    print(cond.shape)



    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        tex_pre = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[1,i, j] >= cfg.side_vertex_pixel_threshold:
                if y[2,i, j] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[2,i, j] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        im.save(os.path.join(save_path, str(numb) + '_act.jpg') )
        quad_draw = ImageDraw.Draw(quad_im)
        tex_pre_draw = ImageDraw.Draw(tex_pre)
        txt_items = []
        boxes = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):

            if np.amin(score) > 0:



                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                boxes.append(rescaled_geo_list)
                box = rescaled_geo_list

                imge = np.array(Image.open(img_path))

                img_tem = Image.fromarray(imge[int(min(box[1], box[3], box[5], box[7])):int(max(box[1], box[3], box[5], box[7])),
                             int(min(box[0], box[2], box[4], box[6])):int(max(box[0], box[2], box[4], box[6])), :])

                probability1 = siamodel.detect_image(image, img_tem)
                probability2 = siamodel.detect_image(image1, img_tem)

                tex_pre_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')

                if probability1 >= 0.6:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='red')
                    quad_draw.text(tuple(geo[1]), 'Similarity:%.3f' % probability1, fill=(255, 0, 0), font=None)

                elif probability2 >= 0.6:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='red')
                    quad_draw.text(tuple(geo[1]), 'Similarity:%.3f' % probability2, fill=(255, 0, 0), font=None)

                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        quad_im.save(os.path.join(save_path, str(numb) + '.jpg'))
        tex_pre.save(os.path.join(save_path, str(numb) + '_pre.jpg'))
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(img_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)

    return boxes, np.array(Image.open(img_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='./CSVTR/ChineseRetrievalCollection/7天连锁酒店/Google_0004.jpeg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

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

def detect_bar(img_bar):

    # 转为灰度图像
    gray = cv2.cvtColor(img_bar, cv2.COLOR_BGR2GRAY)
    barcodes = []
    barcodes = pyzbar.decode(gray)
    #print(barcodes is None)
    #print(len(barcodes))
    if len(barcodes) == 0:
        #print('none')
        return img_bar, 'False', "", ""
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img_bar, (x, y), (x + w, y + h), (255, 255, 0), 4)

        top_left = {"x":x, "y":y}
        width_height = {"w":w, "h":h}


        # 条形码数据为字节对象，所以如果我们想在输出图像上
        #  画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        # 绘出图像上条形码的数据和条形码类型
        barcodeType = barcode.type

        frame = cv2.putText(img_bar, barcodeData, (x, y - 25), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), 4)
        return frame, barcodeData, top_left, width_height

def det_health_tag(img_path, img, tag_path = 'healthtm.png'):
    MIN_MATCH_COUNT = 20
    # img1 = cv2.imread('test2.png', 1)
    img1 = cv2.imread(tag_path, 1)

    # img2 = cv2.imread('2.jpg',1)
    img2 = cv2.imread(img_path, 1)
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
        x_min = int(min(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0]))
        x_max = int(max(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0]))
        y_min = int(min(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1]))
        y_max = int(max(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1]))
        top_left = {"x" : x_min, "y" : y_min}
        width_height = {"w" : x_max - x_min, "h" : y_max - y_min}



        # 根据四个顶点坐标位置在img2图像画出变换后的边框
        img2 = cv2.polylines(img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        flag = 1

    else:
        matchesMask = None
        top_left = None
        width_height = None
    '''
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        return img,flag
    '''
    # 显示匹配结果
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    #plt.figure(figsize=(20, 20))
    #plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    #plt.show()
    #img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img, flag, top_left, width_height

def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))

    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 针对2007年VOC，使用的11个点计算AP，现在不使用
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  # [0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
        mpre = np.concatenate(([0.], prec, [0.]))  # [0.  1.,     0.6666, 0.4285, 0.3043,  0.]

        # compute the precision envelope
        # 计算出precision的各个断点(折线点)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # [1.     1.     0.6666 0.4285 0.3043 0.    ]

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  # precision前后两个值不一样的点
        print(mrec[1:], mrec[:-1])
        print(i)  # [0, 1, 3, 4, 5]

        # AP= AP1 + AP2+ AP3+ AP4
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def recog_label_before(img_path1, img_save_path):
    orientations = 9
    pixels_per_cell = (2, 2)
    cells_per_block = (2, 2)
    threshold = .3
    model_hog = joblib.load('model_name1.npy')

    #ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    np_images = [cv2.imread(img_path1)]
    # print(img_path1)
    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.5)  # 识别中文文本置信度的阈值；

    draw = Image.open(img_path1).copy()

    img_cv1 = cv2.imread(img_path1)
    img_cv2 = cv2.imread(img_path1)
    #img_cv1 = cv2.cvtColor(img_cv1, cv2.COLOR_BGR2RGB)
    dict_result = {'商标':'false', '经销企业':'false', '不适宜人群':'false', '许可证编号':'false', '执行标准':'false', 'r标':'false'}
    dict_result_content = {'商标':'false', '经销企业':'false', '不适宜人群':'false', '许可证编号':'false', '执行标准':'false', 'r标':'false'}
    for result in results:
        data = result['data']
        save_path = result['save_path']

        for infomation in data:

            text = np.array(infomation['text_box_position'])
        
            flag_P = 0
            flag_t = 0
            flag_comp = 0
            box1 = text.reshape(8, )

            box = infomation['text_box_position']
            img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(
                max(box[0][1], box[1][1], box[2][1], box[3][1])),
                          int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(
                              max(box[0][0], box[1][0], box[2][0], box[3][0])), :]
            img_cv2 = cv2.line(img_cv2, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
            img_cv2 = cv2.line(img_cv2, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
            img_cv2 = cv2.line(img_cv2, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
            img_cv2 = cv2.line(img_cv2, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)

            if '经销企业' in infomation['text']:
                img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                img_cv1 = cv2.putText(img_cv1, 'agent:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)
                dict_result['经销企业'] = 'true'
                dict_result_content['经销企业'] = infomation['text']

            if '不适宜人群' in infomation['text']:
                img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                img_cv1 = cv2.putText(img_cv1, 'unapplicable:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)

                dict_result['不适宜人群'] = 'true'

            if '许可证编号' in infomation['text']:
                img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                img_cv1 = cv2.putText(img_cv1, 'QW:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)
                dict_result['许可证编号'] = 'true'
                dict_result_content['许可证编号'] = infomation['text']

            if '执行标准' in infomation['text']:
                img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                img_cv1 = cv2.putText(img_cv1, 'SC:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)
                dict_result['执行标准'] = 'true'
                dict_result_content['执行标准'] = infomation['text']


            img_detect_hog = Image.fromarray(img_detect1)
            img_detect_hog = img_detect_hog.resize((128, 64))
            # print(img.mode)
            gray = img_detect_hog.convert('L')  # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2',
                     feature_vector=True)  # fd= feature descriptor
            fds = fd.reshape(1, -1)  # re shape the image to make a silouhette of hog
            pred = model_hog.predict(
                fds)  # use the SVM model to make a prediction on the HOG features extracted from the window
            if '南京同仁堂' in infomation['text']:
                #if pred == 1:
                if True:
                    if model_hog.decision_function(
                            fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6

                        img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                        img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                        img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                        img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                        img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
                                              (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                                              1, (255, 0, 0), 2)
                        dict_result['商标'] = 'true'
            if pred == 1:
                if model_hog.decision_function(
                        fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6

                    img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                    img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                    img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                    img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                    img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
                                          (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                                          1, (255, 0, 0), 2)
                    dict_result['商标'] = 'true'

            draw = draw.convert('RGB')
            #print(img_save_path)
            cv2.imwrite(os.path.join(img_save_path, '12.png'), img_cv1)
            cv2.imwrite(os.path.join(img_save_path, '12_detect.png'), img_cv2)
            with open("./pdf/test.txt", "w") as f:
                for k,v in dict_result.items():
                    print('%s, %s' %(k,v),sep='',file=f)
                    #f.write(str(dict_result))  # 这句话自带文件关闭功能，不需要再写f.close()
            cv2.imwrite(os.path.join('/home/peng/Downloads/pytorch_AdvancedEast-master', '12.png'), img_cv1)

def tes():
    f = open('word.txt')
    result = list()
    for line in f.readlines():  #
        line = line.strip()  #
        if not len(line):  #
            continue
        result.append(line)
    return result

def pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height):
    result_dict = {}
    result_dict.update({"key":key,"stringvalue":stringvalue,"imagevalue":imagevalue,
                        "relative_jpg":relative_jpg,"top_left":top_left,"width_height":width_height})
    return result_dict

def cap_img(img, top_left, bottom_right, filename):
    image = img.copy()
    image = image[top_left["y"]:bottom_right["y"],top_left["x"]:bottom_right["x"]]
    try:
        cv2.imwrite(filename, image)
    except Exception as e:
        pass
    return filename

def cap_logo(img, top_left, bottom_right, filename):
    image = img.copy()
    image = image[top_left["y"]:bottom_right["y"],top_left["x"]:bottom_right["x"]]
    try:
        cv2.imwrite(filename, image)
    except Exception as e:
        pass
    return filename

# def recog_label(img_path1, img_save_path, dict_result, name):
#     orientations = 9
#     pixels_per_cell = (2, 2)
#     cells_per_block = (2, 2)
#     threshold = .3
#     model_hog = joblib.load('model_name1.npy')
#
#     # ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
#     ocr = hub.Module(name="chinese_ocr_db_crnn_server")

#     np_images = [cv2.imread(img_path1)]
#     # print(img_path1)
#     results = ocr.recognize_text(
#         images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
#         use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
#         output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
#         visualization=False,  # 是否将识别结果保存为图片文件；
#         box_thresh=0.5,  # 检测文本框置信度的阈值；
#         text_thresh=0.5)  # 识别中文文本置信度的阈值；
#
#     draw = Image.open(img_path1).copy()
#
#     img_cv1 = cv2.imread(img_path1)
#     img_cv2 = cv2.imread(img_path1)
#
#     # img_cv1 = cv2.cvtColor(img_cv1, cv2.COLOR_BGR2RGB)
#     dict_result_result = { '商标': 'False', '经销企业': 'False', '不适宜人群': 'False', '许可证编号': 'False', '执行标准': 'False',
#                           '条形码': 'False',
#                           '保健品商标': 'False', '成分':'False', '功效':'False', 'r标':'False'}
#     dict_result_content = { '商标': 'False', '经销企业': 'False', '不适宜人群': 'False', '许可证编号': 'False', '执行标准': 'False',
#                            '条形码': 'False',
#                            '保健品商标': 'False', '成分':'False', '功效':'False', 'r标':'False'}
#
#     #dict_result_result['name'] = name
#     #dict_result_content['name'] = name
#
#     if dict_result['条形码'] == 'True':
#         img_cv1, barcodeData = detect_bar(img_cv1)
#         dict_result_result['条形码'] = 'True'
#         dict_result_content['条形码'] = barcodeData
#         #print(barcodeData)
#
#     if dict_result['保健品商标'] == 'True':
#         img_cv1, flag_B = det_health_tag(img_path1, img_cv1)
#         dict_result_result['保健品商标'] = 'True'
#         dict_result_content['保健品商标'] = 'True'
#
#
#
#     for result in results:
#         data = result['data']
#         save_path = result['save_path']
#
#         for infomation in data:
#
#             text = np.array(infomation['text_box_position'])
#
#             flag_P = 0
#             flag_t = 0
#             flag_comp = 0
#             box1 = text.reshape(8, )
#
#             box = infomation['text_box_position']
#             img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(
#                 max(box[0][1], box[1][1], box[2][1], box[3][1])),
#                           int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(
#                               max(box[0][0], box[1][0], box[2][0], box[3][0])), :]
#             img_cv2 = cv2.line(img_cv2, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#             img_cv2 = cv2.line(img_cv2, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#             img_cv2 = cv2.line(img_cv2, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#             img_cv2 = cv2.line(img_cv2, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#
#             if dict_result['经销企业'] == 'True':
#                 if '经销企业' in infomation['text']:
#                     img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                     img_cv1 = cv2.putText(img_cv1, 'agent:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                           1, (255, 0, 0), 4)
#                     dict_result_result['经销企业'] = 'True'
#                     dict_result_content['经销企业'] = infomation['text']
#
#             if dict_result['不适宜人群'] == 'True':
#                 if '不适宜人群' in infomation['text']:
#                     img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                     img_cv1 = cv2.putText(img_cv1, 'unapplicable:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                           1, (255, 0, 0), 4)
#
#                     dict_result_result['不适宜人群'] = 'True'
#                     dict_result_content['不适宜人群'] = infomation['text']
#
#             if dict_result['许可证编号'] == 'True':
#                 if '许可证编号' in infomation['text']:
#                     img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                     img_cv1 = cv2.putText(img_cv1, 'QW:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                           1, (255, 0, 0), 4)
#                     dict_result_result['许可证编号'] = 'True'
#                     dict_result_content['许可证编号'] = infomation['text']
#
#             if dict_result['执行标准'] == 'True':
#                 if '执行标准' in infomation['text']:
#                     img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                     img_cv1 = cv2.putText(img_cv1, 'SC:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                           1, (255, 0, 0), 4)
#                     dict_result_result['执行标准'] = 'True'
#                     dict_result_content['执行标准'] = infomation['text']
#
#             if dict_result['功效'] == 'True':
#                 result_usage = tes()
#                 for use_name in result_usage:
#                     if use_name in infomation['text']:
#                         img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 0), 4)
#                         img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (0, 0, 0), 4)
#                         img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (0, 0, 0), 4)
#                         img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (0, 0, 0), 4)
#                         img_cv1 = cv2.putText(img_cv1, 'usage', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                               1, (255, 0, 0), 4)
#                         dict_result_result['功效'] = 'True'
#                         dict_result_content['功效'] = infomation['text']
#
#
#             if dict_result['成分'] == 'True':
#                 if '成分' in infomation['text']:
#                     img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                     img_cv1 = cv2.putText(img_cv1, 'Component', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                           1, (255, 0, 0), 4)
#                     dict_result_result['成分'] = 'True'
#                     dict_result_content['成分'] = infomation['text']
#
#
#
#             img_detect_hog = Image.fromarray(img_detect1)
#             img_detect_hog = img_detect_hog.resize((128, 64))
#             # print(img.mode)
#             gray = img_detect_hog.convert('L')  # convert the image into single channel i.e. RGB to grayscale
#             # calculate HOG for positive features
#             fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2',
#                      feature_vector=True)  # fd= feature descriptor
#             fds = fd.reshape(1, -1)  # re shape the image to make a silouhette of hog
#             pred = model_hog.predict(
#                 fds)  # use the SVM model to make a prediction on the HOG features extracted from the window
#             if '南京同仁堂' in infomation['text']:
#                 # if pred == 1:
#                 if True:
#                     if model_hog.decision_function(
#                             fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
#
#                         img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                         img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                         img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                         img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                         img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
#                                               (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
#                                               1, (255, 0, 0), 4)
#                         dict_result['商标'] = 'True'
#                         dict_result_content['商标'] = infomation['text']
#             if pred == 1:
#                 if model_hog.decision_function(
#                         fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
#
#                     img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
#                     img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
#                     img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
#                                           (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
#                                           1, (255, 0, 0), 4)
#                     dict_result_result['商标'] = 'True'
#                     dict_result_content['商标'] = infomation['text']
#
#
#
#             draw = draw.convert('RGB')
#             # print(img_save_path)
#
#     if dict_result['r标'] == 'True':
#         #img_cv1, flag_C = det_health_tag(img_path1, img_cv1, tag_path='r.png')
#         #img_cv1 = detect_r(img_cv1)
#         flag_rb = det_r(img_cv1)
#         if flag_rb != 0:
#             dict_result_result['r标'] = 'True'
#             dict_result_content['r标'] = 'R标'
#
#     cv2.imwrite(os.path.join(img_save_path, '{}.png'.format(name) ), img_cv1)
#     cv2.imwrite(os.path.join(img_save_path, '{}_detect.png'.format(name) ), img_cv2)
#     file = open("./pdf/txt/{}.txt".format(name), "w").close()
#     final_result = {'name': name}
#     with open("./pdf/txt/{}.txt".format(name), "a") as f:
#         for k, v in dict_result_result.items():
#             for q, w in dict_result_content.items():
#                 if q == k:
#                     print('%s, %s, %s' % (k, v, w), sep='', file=f)
#                     final_result[k] = [v, w]
#             # f.write(str(dict_result))  # 这句话自带文件关闭功能，不需要再写f.close()
#
#
#     cv2.imwrite(os.path.join('/home/peng/Downloads/pytorch_AdvancedEast-master', '123.png'), img_cv1)
#     img_res = img_resize(img_cv1)
#     cv2.imwrite('12.png', img_res)
#     # cv2.imwrite(os.path.join('/home/peng/Downloads/pytorch_AdvancedEast-master', '12.png'), img_resize)
#
#     return final_result

def recog_label(recognition, img_path1, text_path, img_save_path, name, logo_file):
    orientations = 9
    pixels_per_cell = (2, 2)
    cells_per_block = (2, 2)
    threshold = .3
    model_hog = joblib.load('model_name1.npy')

    # ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")

    # src_image = cv2.imread(img_path1)
    #
    # binary_image = getBinary(img_path1)
    # image_in = removeContours(binary_image)
    # ret, image_in = cv2.threshold(image_in, 100, 255, cv2.THRESH_BINARY)
    # vermean, verlist = verProject(image_in)
    # hormean, horlist = horProject(image_in)

    # w_start, w_end, h_start, h_end = roi(verlist, horlist, vermean, hormean)
    #
    # image_result = src_image[h_start:h_end, w_start:w_end]

    np_images = [cv2.imread(img_path1)]
    for file in os.listdir(text_path):
        filename = os.path.join(text_path, file)
        np_images.append(cv2.imread(filename, 1))

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='/pack/temp/ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.5)  # 识别中文文本置信度的阈值；

    draw = Image.open(img_path1).copy()

    final_result = []

    img_cv1 = cv2.imread(img_path1)
    img_cv2 = cv2.imread(img_path1)

    # img_cv1 = cv2.cvtColor(img_cv1, cv2.COLOR_BGR2RGB)
    dict_result_result = recognition
    dict_result_content = recognition
    recognition_find = []

    #dict_result_result['name'] = name
    #dict_result_content['name'] = name

    #identify barcode
    if (dict_result_content.get("barcode", 0) != 0):
        img_cv1, barcodeData, top_left, width_height = detect_bar(img_cv1)
        if(barcodeData != "False"):
            key = "barcode"
            relative_jpg = [{"logo_detect_path":os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            stringvalue = barcodeData
            bottom_right = {"x":top_left["x"] + width_height["w"], "y":top_left["y"] + width_height["h"]}
            dst = "/pack/output/barcode"
            barcode_filename = os.path.join(dst, '{}_barcode.png'.format(name))
            barcode_path = cap_img(img_cv1, top_left, bottom_right, barcode_filename)
            imagevalue = barcode_path
            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)

            recognition_find.append("barcode")

    # identify healthtm
    if (dict_result_content.get("healthtm", 0) != 0):
        img_cv1, flag_B, top_left, width_height = det_health_tag(img_path1, img_cv1)
        if(flag_B != 0):
            key = "healthtm"
            stringvalue = True
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            bottom_right = {"x": (top_left["x"] + width_height["w"]), "y": (top_left["y"] + width_height["h"])}
            dst = "/pack/output/healthtm"
            healthtm_filename = os.path.join(dst, '{}_healthtm.png'.format(name))
            healthtm_path = cap_img(img_cv1, top_left, bottom_right, healthtm_filename)
            imagevalue = healthtm_path

            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)

            recognition_find.append("healthtm")

    # #identify logo
    if (dict_result_content.get("logo_trt", 0) != 0):
        img_cv1, flag_B, top_left, width_height = det_health_tag(img_path1, img_cv1, tag_path = logo_file["logo_trt"])
        if (flag_B != 0):
            key = "logo_trt"
            stringvalue = True
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            bottom_right = {"x": top_left["x"] + width_height["w"], "y": top_left["y"] + width_height["h"]}
            dst = "/pack/output/logo"
            trt_filename = os.path.join(dst, '{}_logo_trt.png'.format(name))
            trt_path = cap_logo(img_cv1, top_left, bottom_right, trt_filename)
            imagevalue = trt_path

            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)

            recognition_find.append("logo_trt")

    if (dict_result_content.get("logo_yjlp", 0) != 0):
        img_cv1, flag_B, top_left, width_height = det_health_tag(img_path1, img_cv1, tag_path = logo_file["logo_yjlp"])
        if (flag_B != 0):
            key = "logo_yjlp"
            stringvalue = True
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            bottom_right = {"x": top_left["x"] + width_height["w"], "y": top_left["y"] + width_height["h"]}
            dst = "/pack/output/logo"
            yjlp_filename = os.path.join(dst, '{}_logo_yjlp.png'.format(name))
            yjlp_path = cap_logo(img_cv1, top_left, bottom_right, yjlp_filename)
            imagevalue = yjlp_path

            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)

            recognition_find.append("logo_yjlp")

    if (dict_result_content.get("logo_lhqk", 0) != 0):
        img_cv1, flag_B, top_left, width_height = det_health_tag(img_path1, img_cv1, tag_path = logo_file["logo_lhqk"])
        if (flag_B != 0):
            key = "logo_lhqk"
            stringvalue = True
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            bottom_right = {"x": top_left["x"] + width_height["w"], "y": top_left["y"] + width_height["h"]}
            dst = "/pack/output/logo"
            lhqk_filename = os.path.join(dst, '{}_logo_lhqk.png'.format(name))
            lhqk_path = cap_logo(img_cv1, top_left, bottom_right, lhqk_filename)
            imagevalue = lhqk_path

            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)

            recognition_find.append("logo_lhqk")

    if (dict_result_content.get("logo_yjxl", 0) != 0):
        img_cv1, flag_B, top_left, width_height = det_health_tag(img_path1, img_cv1, tag_path = logo_file["logo_yjxl"])
        if (flag_B != 0):
            key = "logo_yjxl"
            stringvalue = True
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            bottom_right = {"x": top_left["x"] + width_height["w"], "y": top_left["y"] + width_height["h"]}
            dst = "/pack/output/logo"
            yjxl_filename = os.path.join(dst, '{}_logo_yjxl.png'.format(name))
            yjxl_path = cap_logo(img_cv1, top_left, bottom_right, yjxl_filename)
            imagevalue = yjxl_path

            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)

            recognition_find.append("logo_yjxl")

    # identify text
    for result in results:
        data = result['data']
        save_path = result['save_path']

        for infomation in data:

            text = np.array(infomation['text_box_position'])

            flag_P = 0
            flag_t = 0
            flag_comp = 0
            box1 = text.reshape(8, )

            box = infomation['text_box_position']
            img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(
                max(box[0][1], box[1][1], box[2][1], box[3][1])),
                          int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(
                              max(box[0][0], box[1][0], box[2][0], box[3][0])), :]
            img_cv2 = cv2.line(img_cv2, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
            img_cv2 = cv2.line(img_cv2, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
            img_cv2 = cv2.line(img_cv2, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
            img_cv2 = cv2.line(img_cv2, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)


            for k, v in recognition.items():
                if(k != "barcode" and k != "healthtm" and k != "r标" and k != "trademark"):
                    if v in infomation['text']:
                        img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
                        img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
                        img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
                        img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
                        img_cv1 = cv2.putText(img_cv1, 'agent:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                              1, (255, 0, 0), 4)

                        key = k
                        stringvalue = infomation['text']
                        imagevalue = ""
                        relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                                         "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
                        top_left = ""
                        width_height = ""
                        #top_left = {"x": int(box1[0]), "y": int(box1[1])}
                        #width_height = {"w":int(box1[2]-box1[0]), "h":int(box1[7]-box1[1])}
                        #bottom_right = {"x": top_left["x"] + width_height["w"], "y": top_left["y"] + width_height["h"]}
                        #dst = "/pack/output/text"
                        #text_filename = os.path.join(dst, '{}_{}.png'.format(name, k))
                        #text_path = cap_img(img_cv1, top_left, bottom_right, text_filename)
                        #imagevalue = text_path
                        result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
                        final_result.append(result_dict)

                        recognition_find.append(k)

            #identify trademark
            if (dict_result_content.get("trademark", 0) != 0):
                img_detect_hog = Image.fromarray(img_detect1)
                img_detect_hog = img_detect_hog.resize((128, 64))
                # print(img.mode)
                gray = img_detect_hog.convert('L')  # convert the image into single channel i.e. RGB to grayscale
                # calculate HOG for positive features
                fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2',
                         feature_vector=True)  # fd= feature descriptor
                fds = fd.reshape(1, -1)  # re shape the image to make a silouhette of hog
                pred = model_hog.predict(
                    fds)  # use the SVM model to make a prediction on the HOG features extracted from the window
                #if '南京同仁堂' in infomation['text']:
                    # if pred == 1:
                    # if True:
                    #     if model_hog.decision_function(
                    #             fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6

                            # img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
                            # img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
                            # img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
                            # img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
                            # img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
                            #                       (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                            #                       1, (255, 0, 0), 4)

                            # key = "trt"
                            # stringvalue = infomation['text']
                            # imagevalue = ""
                            # relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                            #                  "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
                            # top_left = {"x": int(box1[0]), "y": int(box1[1])}
                            # width_height = {"w":int(box1[2]-box1[0]), "h":int(box1[7]-box1[1])}
                            # result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
                            # final_result.append(result_dict)

                if pred == 1:
                    if model_hog.decision_function(
                            fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6

                        img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 4)
                        img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 4)
                        img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 4)
                        img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 4)
                        img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
                                              (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                                              1, (255, 0, 0), 4)

                        key = "trademark"
                        stringvalue = infomation['text']
                        relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                                         "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
                        top_left = {"x": int(box1[0]), "y": int(box1[1])}
                        width_height = {"w": int(box1[2] - box1[0]), "h": int(box1[7] - box1[1])}
                        bottom_right = {"x": top_left["x"] + width_height["w"], "y": top_left["y"] + width_height["h"]}
                        dst = "/pack/output/trademark"
                        mark_filename = os.path.join(dst, '{}_{}.png'.format(name, key))
                        mark_path = cap_img(img_cv1, top_left, bottom_right, mark_filename)
                        imagevalue = mark_path
                        result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)

                        final_result.append(result_dict)
                        recognition_find.append("trademark")


                draw = draw.convert('RGB')
            # print(img_save_path)

    if (dict_result_content.get("r标", 0) != 0):
        #img_cv1, flag_C = det_health_tag(img_path1, img_cv1, tag_path='r.png')
        #img_cv1 = detect_r(img_cv1)
        flag_rb = det_r(img_cv1)
        if flag_rb != 0:
            key = "r标"
            stringvalue = True
            imagevalue = ""
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            top_left = None
            width_height = None
            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)
            recognition_find.append("r标")

    for k, v in recognition.items():
        if(k not in recognition_find):
            key = k
            stringvalue = False
            imagevalue = None
            relative_jpg = [{"logo_detect_path": os.path.join(img_save_path, '{}.png'.format(name)),
                             "text_detect_path": os.path.join(img_save_path, '{}_detect.png'.format(name))}]
            top_left = None
            width_height = None
            result_dict = pack_dict(key, stringvalue, imagevalue, relative_jpg, top_left, width_height)
            final_result.append(result_dict)


    cv2.imwrite(os.path.join(img_save_path, '{}.png'.format(name) ), img_cv1)
    cv2.imwrite(os.path.join(img_save_path, '{}_detect.png'.format(name) ), img_cv2)


    #file = open("./pdf/txt/{}.txt".format(name), "w").close()
    #final_result = {'name': name}
    # with open("./pdf/txt/{}.txt".format(name), "a") as f:
    #     for k, v in dict_result_result.items():
    #         for q, w in dict_result_content.items():
    #             if q == k:
    #                 print('%s, %s, %s' % (k, v, w), sep='', file=f)
    #                 final_result[k] = [v, w]
            # f.write(str(dict_result))  # 这句话自带文件关闭功能，不需要再写f.close()

    #cv2.imwrite(os.path.join('/home/trt_pro/pytorch_AdvancedEast-master', '123.png'), img_cv1)
    #img_res = img_resize(img_cv1)
    #cv2.imwrite('12.png', img_res)
    # cv2.imwrite(os.path.join('/home/peng/Downloads/pytorch_AdvancedEast-master', '12.png'), img_resize)

    return final_result

if __name__ == '__main__':
    img_path1 = '.jpg'
    img_cv1 = cv2.imread(img_path1)
    img_cv1, flag_B, top_left, width_height = det_health_tag(img_path1, img_cv1, 'r.png')
    cv2.imshow("img", img_cv1)
    cv2.waitKey(0)

# if __name__ == '__main__':
#     config, args = parse_args()
#     img_path = args.path
#     threshold = float(args.threshold)
#     print(img_path, threshold)
#     model_path='./saved_model/mb3_512_model_epoch_535.pth'
#
#     image = 'img/11.jpg'
#     image1 = 'img/144.jpg'
#     image2 = 'img/55.jpg'
#     # 加载移动端预训练模型
#     #ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
#     ocr = hub.Module(name="chinese_ocr_db_crnn_server")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = EAST().to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     model_reg = crnn.get_crnn(config).to(device)
#     print('loading pretrained model from {0}'.format(args.checkpoint))
#     checkpoint = torch.load(args.checkpoint)
#     if 'state_dict' in checkpoint.keys():
#         model_reg.load_state_dict(checkpoint['state_dict'])
#     else:
#         model_reg.load_state_dict(checkpoint)
#
#     model_hog = joblib.load('model_name1.npy')
#     #img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂')
#     img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/test')
#     #img_paths = os.path.abspath('./CSVTR/CSVTR')
#     img_files = sorted([os.path.join(img_paths, img_file) for img_file in os.listdir(img_paths)])
#     img_save_path = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/30/')
#     img_save_path2 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/29/')
#     sigmodel = Siamese()
#
#     orientations = 9
#     pixels_per_cell = (2, 2)
#     cells_per_block = (2, 2)
#     threshold = .3
#     #conf = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#     #conf = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
#     conf = [0.78]
#     recall = []
#     pres = []
#     f1 = []
#     for i in conf:
#         numb = 0
#         numb_d = 0
#         total = 0
#         k = 0
#         acc = 0
#         img_num = 0
#         started = time.time()
#         for img_path1 in img_files:
#             #print(img_path1)
#             img_num = img_num+1
#             #img_path1 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂/10.jpeg')
#             #img_path3 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/1.jpeg')
#             np_images = [cv2.imread(img_path1)]
#             #print(img_path1)
#             results = ocr.recognize_text(
#                 images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
#                 use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
#                 output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
#                 visualization=False,  # 是否将识别结果保存为图片文件；
#                 box_thresh=0.5,  # 检测文本框置信度的阈值；
#                 text_thresh=0.5)  # 识别中文文本置信度的阈值；
#
#             draw = Image.open(img_path1).copy()
#             img_cv = cv2.imread(img_path1)
#             #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#
#             img_cv1 = cv2.imread(img_path1)
#             #img_cv1 = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#             # imge = np.array(Image.open(img_path1))
#             drawim = ImageDraw.Draw(draw)
#
#             or_draw = Image.open(img_path1).copy()
#             or_drawim = ImageDraw.Draw(or_draw)
#
#             numb = numb + 1
#
#             for result in results:
#                 data = result['data']
#                 save_path = result['save_path']
#
#                 for infomation in data:
#
#                     text = np.array(infomation['text_box_position'])
#
#                     flag_P = 0
#                     flag_t = 0
#                     flag_comp = 0
#                     box1 = text.reshape(8,)
#
#                     box = infomation['text_box_position']
#                     img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(
#                         max(box[0][1], box[1][1], box[2][1], box[3][1])),
#                                   int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(
#                                       max(box[0][0], box[1][0], box[2][0], box[3][0])), :]
#
#
#                     if  '经销企业' in infomation['text']:
#                         img_cv1 = cv2.putText(img_cv1, '经销企业:ture' , (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                              1, (255, 0, 0), 2)
#
#                     if '生产商' in infomation['text']:
#                         img_cv1 = cv2.putText(img_cv1, '生产商:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                               1, (255, 0, 0), 2)
#
#                     img_cv = cv2.line(img_cv, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
#                     img_cv = cv2.line(img_cv, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 2)
#                     img_cv = cv2.line(img_cv, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 2)
#                     img_cv = cv2.line(img_cv, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 2)
#
#                     if infomation['text'] == '南京同仁堂':
#                         total = total + 1
#                         flag_t = 1
#
#                         img_cv = cv2.line(img_cv, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
#                         img_cv = cv2.line(img_cv, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 2)
#                         img_cv = cv2.line(img_cv, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 2)
#                         img_cv = cv2.line(img_cv, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 2)
#                         img_cv = cv2.putText(img_cv, 'numb:%.3f' % total, (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
#                                                  1, (255, 0, 0), 2)
#                         # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
#
#                     '''
#                     img_detect_hog = Image.fromarray(img_detect1)
#
#                     img_detect_hog = img_detect_hog.resize((128, 64))
#                     # print(img.mode)
#                     gray = img_detect_hog.convert('L')  # convert the image into single channel i.e. RGB to grayscale
#                     # calculate HOG for positive features
#                     fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2',
#                              feature_vector=True)  # fd= feature descriptor
#                     fds = fd.reshape(1, -1)  # re shape the image to make a silouhette of hog
#                     pred = model_hog.predict(
#                         fds)  # use the SVM model to make a prediction on the HOG features extracted from the window
#                     '''
#                     probability1 = sigmodel.detect_image(Image.open(image), Image.fromarray(img_detect1))
#                     probability2 = sigmodel.detect_image(Image.open(image1), Image.fromarray(img_detect1))
#
#                     if probability1 >= i:
#                         img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
#                         img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 2)
#                         img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 2)
#                         img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 2)
#                         img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % probability1,
#                                               (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
#                                               1, (255, 0, 0), 2)
#                         flag_P = 1
#
#                         # print([box1[0], box1[1], box1[2], box1[3], box1[4], box1[5], box1[6], box1[7],box1[0], box1[1]])
#
#                     elif probability2 >= i:
#                         img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
#                         img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 2)
#                         img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 2)
#                         img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 2)
#                         img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % probability2,
#                                               (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
#                                               1, (255, 0, 0), 2)
#
#                         # print([box1[0], box1[1], box1[2], box1[3], box1[4], box1[5], box1[6], box1[7], box1[0], box1[1]])
#                         flag_P = 1
#                     '''
#                     if pred == 1:
#                         if model_hog.decision_function(
#                             fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
#
#                             img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
#                             img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
#                             img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
#                             img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
#                             img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds), (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
#                                                  1, (255, 0, 0), 2)
#
#                             flag_P = 1
#                     '''
#                     if flag_P == 1 and flag_t == 1:
#                         acc = acc + 1
#                     if flag_P != 0:
#                         k = k + 1
#                         flag_P = 0
#                     #print(probability1, probability2, k)
#                     draw = draw.convert('RGB')
#                     #draw.save(os.path.join(img_save_path, str(numb) + '.jpg'))
#                     cv2.imwrite(os.path.join(img_save_path, str(numb) + '.jpg'), img_cv1)
#                     cv2.imwrite(os.path.join(img_save_path2, str(numb) + '_1.jpg'), img_cv)
#                     or_draw = or_draw.convert('RGB')
#
#         recall.append(acc/total)
#         pres.append(acc/max(k, 0.1))
#         f1.append((2*(acc/total)*(acc/max(k, 0.1)))/max(((acc/total)+(acc/max(k, 0.1))), 0.1))
#         print(recall,pres,f1,acc,total,k)
#
#         '''
#         #print(voc_ap(np.array(pres), np.array(recall)))
#
#         ###shengcheng xunlian ji
#
#                 for infomation in data:
#                     print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'],
#                           '\ntext_box_position: ', infomation['text_box_position'])
#                     if infomation['text'] == '南京同仁堂':
#                         numb = numb + 1
#                         box = infomation['text_box_position']
#                         print(box[0][1])
#                         img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(max(box[0][1], box[1][1], box[2][1], box[3][1])), int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(max(box[0][0], box[1][0], box[2][0], box[3][0])), :]
#                         Img = Image.fromarray(img_detect1)
#                         Img.save(os.path.join(img_save_path, str(numb) + '.jpg'))
#
#         #break
#         '''
#
#         finished = time.time()
#         print(img_num)
#         print('elapsed time: {0}'.format((finished - started)))
