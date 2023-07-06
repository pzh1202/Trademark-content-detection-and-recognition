# -*- coding: utf-8 -*-
"""
Created on ：2018/04/03
@author: Freeman
"""
import os

import cv2
from PyPDF2 import PdfFileReader, PdfFileWriter
import fitz
__all__ = ['PDFMake']


class PDFMake(object):
    def __init__(self, outputdirectory=''):
        self.outputdirectory = outputdirectory
        #self.outputdirectory_boost = outputdirectory_boost

    def __check_page_list(self):
        # 检测是否为string类型
        if type(self.listmode) != type('0-6,8-9'):
            return True
        else:
            # 检测是否含有非法字符
            for i in range(1, len(self.listmode) - 1):
                if self.listmode[i] not in \
                        ['-', ',', '\'', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    return True
        return False

    def spilt_pdf(self,  inputfilename=None, listmode=None, name = '' ):
        self.listmode = listmode
        # 读取PDF文件
        pdfFile = PdfFileReader(open(inputfilename, "rb"))
        pageCount = pdfFile.getNumPages()


        if self.listmode != None:
            for i in range(int(self.listmode)):
                pdfWriter = PdfFileWriter()
                page = pdfFile.getPage(i)
                pdfWriter.addPage(page)
                pdfWriterNmae = '/home/trt/trt_pro/pdf/temp-pdf' +'/{}{}'.format(name,i+1) + '.pdf'
                output = self.outputdirectory +'/{}{}'.format(name, i+1) + '.jpg'

                pdfWriter.write(open(pdfWriterNmae, 'wb'))
                doc = fitz.open(pdfWriterNmae)
                page = doc.loadPage(0)  # PDF页数
                pix = page.getPixmap()

                output_ori = '/pack/temp/ori' +'/{}'.format(name) + '.jpg'
                pix.writePNG(output_ori)  # 保存
                im_jpg = cv2.imread(output_ori, 1)
                if(im_jpg.shape[0]*im_jpg.shape[1] < 500000):
                    zoom_x = 4
                    zoom_y = 4
                    mat = fitz.Matrix(zoom_x, zoom_y)
                    pix = page.getPixmap(matrix=mat)
                    pix.writePNG(output)  # 保存
                    os.remove(output_ori)
                elif(500000 < im_jpg.shape[0]*im_jpg.shape[1] < 1000000):
                    zoom_x = 3
                    zoom_y = 3
                    mat = fitz.Matrix(zoom_x, zoom_y)
                    pix = page.getPixmap(matrix=mat)
                    pix.writePNG(output)  # 保存
                    os.remove(output_ori)
                elif(1000000 < im_jpg.shape[0]*im_jpg.shape[1] < 2000000):
                    zoom_x = 2
                    zoom_y = 2
                    mat = fitz.Matrix(zoom_x, zoom_y)
                    pix = page.getPixmap(matrix=mat)
                    pix.writePNG(output)  # 保存
                    os.remove(output_ori)
                elif(im_jpg.shape[0]*im_jpg.shape[1] > 40000000):
                    zoom_x = 0.6
                    zoom_y = 0.6
                    mat = fitz.Matrix(zoom_x, zoom_y)
                    pix = page.getPixmap(matrix=mat)
                    pix.writePNG(output)  # 保存
                    os.remove(output_ori)
                else:
                    zoom_x = 1
                    zoom_y = 1
                    mat = fitz.Matrix(zoom_x, zoom_y)
                    pix = page.getPixmap(matrix=mat)
                    pix.writePNG(output)
                    os.remove(output_ori)

        else:
            if self.__check_page_list():
                raise ValueError
            else:
                pdfWriter = PdfFileWriter()
                part = self.listmode.split(',')
                for k in part:
                    start = int(k.split('-')[0])
                    end = int(k.split('-')[1])
                    for m in range(start - 1, end):
                        page = pdfFile.getPage(m)
                        pdfWriter.addPage(page)
                pdfWriterNmae = self.outputdirectory + '/part ' + self.listmode + '.pdf'
                pdfWriter.write(open(pdfWriterNmae, 'wb'))

    def merge_pdf(self, flienamelist=None):
        print(flienamelist)
        pdfWriter = PdfFileWriter()
        for i in flienamelist:
            pdfFile = PdfFileReader(open(i, "rb"))
            pdfWriter.appendPagesFromReader(pdfFile)
        pdfWriterNmae = self.outputdirectory + '/mergeByMaster.pdf'
        pdfWriter.write(open(pdfWriterNmae, 'wb'))
