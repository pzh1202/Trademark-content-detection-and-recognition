import argparse

import torch
from torchvision import transforms
from model import EAST
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw

import cfg

from preprocess import resize_image
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
from ap import voc_ap
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def load_pil(img):
    '''convert PIL Image to torch.Tensor
    '''
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)



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


def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')




if __name__ == '__main__':
    config, args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)
    model_path = './saved_model/mb3_512_model_epoch_535.pth'

    image = 'img/11.jpg'
    image1 = 'img/144.jpg'
    image2 = 'img/55.jpg'
    # 加载移动端预训练模型
    # ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tyle_img = 'img/test/'
    model_reg = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model_reg.load_state_dict(checkpoint['state_dict'])
    else:
        model_reg.load_state_dict(checkpoint)
    name_paths = ('./CSVTR/CSVTR')
    name_files = sorted([name_file for name_file in os.listdir(name_paths)])
    name_files =['GUCCI']
    ap = []
    for name in name_files:
        print(name)
        # img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂')
        #img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/test')
        #img_paths = os.path.abspath(os.path.join('./CSVTR/CSVTR/', name))
        img_paths = os.path.abspath(os.path.join('./CSVTR/', name))
        img_files = sorted([os.path.join(img_paths, img_file) for img_file in os.listdir(img_paths)])
        print(img_files)
        #img_save_path = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/30/')
        img_save_path2 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/29/')
        #img_save_path = os.path.abspath(os.path.join('./CSVTR/transresult/', name + '/'))
        img_save_path = os.path.abspath(os.path.join('./CSVTR/paper/', name + '/'))
        sigmodel = Siamese()
        image = os.path.join(tyle_img, name + '.jpg')
        orientations = 9
        pixels_per_cell = (2, 2)
        cells_per_block = (2, 2)
        threshold = .3
        conf = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #conf = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        #conf = [0.75]
        recall = []
        pres = []
        f1 = []
        #conf = [0.73]
        for i in conf:
            numb = 0
            numb_d = 0
            total = 0
            k = 0
            acc = 0
            img_num = 0
            started = time.time()
            for img_path1 in img_files:
                # print(img_path1)
                img_num = img_num + 1
                #print(img_path1)
                # img_path1 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂/10.jpeg')
                # img_path3 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/1.jpeg')
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
                img_cv = cv2.imread(img_path1)
                # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                img_cv1 = cv2.imread(img_path1)
                # img_cv1 = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                # imge = np.array(Image.open(img_path1))
                drawim = ImageDraw.Draw(draw)

                or_draw = Image.open(img_path1).copy()
                or_drawim = ImageDraw.Draw(or_draw)

                numb = numb + 1

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



                        if infomation['text'] == name:
                            total = total + 1
                            flag_t = 1

                            img_cv = cv2.line(img_cv, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 6)
                            img_cv = cv2.line(img_cv, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 6)
                            img_cv = cv2.line(img_cv, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 6)
                            img_cv = cv2.line(img_cv, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 6)
                            img_cv = cv2.putText(img_cv, 'numb:%.3f' % total, (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                                 1, (255, 0, 0), 2)
                            # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度

                        probability1 = sigmodel.detect_image(Image.open(image), Image.fromarray(img_detect1))
                        #probability2 = sigmodel.detect_image(Image.open(image1), Image.fromarray(img_detect1))

                        if probability1 >= i:
                            img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 6)
                            img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 6)
                            img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 6)
                            img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 6)
                            #img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % probability1,
                            #                      (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                            #                      1, (255, 0, 0), 2)
                            flag_P = 1

                            # print([box1[0], box1[1], box1[2], box1[3], box1[4], box1[5], box1[6], box1[7],box1[0], box1[1]])
                        '''
                        elif probability2 >= i:
                            img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
                            img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (0, 255, 0), 2)
                            img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (0, 255, 0), 2)
                            img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (0, 255, 0), 2)
                            img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % probability2,
                                                  (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                                                  1, (255, 0, 0), 2)

                            # print([box1[0], box1[1], box1[2], box1[3], box1[4], box1[5], box1[6], box1[7], box1[0], box1[1]])
                            flag_P = 1
                        '''
                        if flag_P == 1 and flag_t == 1:
                            acc = acc + 1
                        if flag_P != 0:
                            k = k + 1
                            flag_P = 0
                        # print(probability1, probability2, k)
                        draw = draw.convert('RGB')
                        # draw.save(os.path.join(img_save_path, str(numb) + '.jpg'))
                        cv2.imwrite(os.path.join(img_save_path, str(numb) + '.jpg'), img_cv1)
                        #cv2.imwrite(os.path.join(img_save_path2, str(numb) + '_1.jpg'), img_cv)
                        or_draw = or_draw.convert('RGB')
            print(img_num)

            recall.append(acc / (total + 0.1))
            pres.append(acc / max(k, 0.1))
            f1.append((2 * (acc / (total + 0.1)) * (acc / max(k, 0.1))) / max(((acc / (total + 0.1)) + (acc / max(k, 0.1))), 0.1))
            print(recall, pres, f1, acc, total, k)
            finished = time.time()
            print(img_num)
            print('elapsed time: {0}'.format((finished - started)))
        ap.append(voc_ap(pres, recall))
        print(ap)
