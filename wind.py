
# -*- coding: utf-8 -*-
from tkinter import *
import tkinter.filedialog
from PdfMake import PDFMake
from predict import recog_label
import os


def getPdfFileName():
    filename = tkinter.filedialog.askopenfilename()
    pdfFileName.set(filename)
    print(filename)
    if filename != '':
        lb1.config(text="识别文件：" + filename)
    else:
        lb1.config(text="您没有选择任何文件")



def selectPath():
    path = tkinter.filedialog.askdirectory()
    outputdirectory.set(path)
    if path != '':
        lb2.config(text="输出路径：" + path)
    else:
        lb2.config(text="您没有选择任何路径")


def Recogpdf():
    if outputdirectory.get() != '' and pdfFileName.get() != '':
        #master = PDFMake(outputdirectory=outputdirectory.get())
        #if pageList.get() == '':
        if True:
            #master.spilt_pdf(pdfFileName.get())
            recog_label(pdfFileName.get(),outputdirectory.get())
            lb3.config(text="是个可靠的正品！")
            '''
            tk = Toplevel()
            tk.title('检测图片')
            #img = os.path.join(outputdirectory.get(), '12.jpg')
            #photo = PhotoImage(file="12.png")
            
            photo = PhotoImage(file="12.png")
            theLabel = Label(tk, image=photo)
            label = Label(tk, text="正品", font=('heiti', 20), fg='red', bg='pink')
            label.place(x=10, y=10)
            theLabel.pack()
            tk.mainloop()
            '''
    else:
        lb3.config(text="拆分失败:选择文件或者目录有误！")

def splitPdf():
    if outputdirectory.get() != '' and pdfFileName.get() != '':
        master = PDFMake(outputdirectory=outputdirectory.get())
        if pageList.get() == '':
            master.spilt_pdf(pdfFileName.get())
            llb3.config(text="拆分成功！")
        else:
            try:
                master.spilt_pdf(pdfFileName.get(), pageList.get())
                llb3.config(text="拆分成功！")
            except ValueError:
                llb3.config(text="拆分失败:ValueError！")
    else:
        llb3.config(text="拆分失败:选择文件或者目录有误！")

if __name__ == '__main__':

    root = tkinter.Tk()
    root.geometry('700x450+300+50')

    outputdirectory = StringVar()
    pdfFileName = StringVar()
    pdfFileNames = StringVar()
    pageList = StringVar()
    var1 = BooleanVar()
    var2 = BooleanVar()
    var3 = BooleanVar()
    var4 = BooleanVar()
    var5 = BooleanVar()

    root.title('label recognize       Copyright © 2021 by PengZhiheng')
    root['bg'] = '#a9ed96'
    root.attributes("-alpha",0.97)

    lb0 = Label(root, text='需要识别或者拆分的文件', width=20, font = 'Helvetica -20 bold',bg='#a9ed96')
    lb0.grid(row=0, columnspan=2,stick=W,pady=10,)

    btn1 = Button(root, text="选择需要识别或者拆分的文件", width=20, command=getPdfFileName)
    btn1.grid(row=1, column=0, stick=W, pady=10, padx=50)

    lb1 = Label(root, text='', width=50)
    lb1.grid(row=1, column=1)

    btn2 = Button(root, text="识别后的存储路径", width=20, command=selectPath)
    btn2.grid(row=2, column=0, stick=W, pady=10,padx=50)

    lb2 = Label(root, text='', width=50)
    lb2.grid(row=2, column=1)

    lbe = Label(root, text='需要识别的内容', width=20)
    lbe.grid(row=3, column=0, stick=W,pady=10,padx=50)

    #ch1 = Checkbutton(root, text='label', variable=var1, onvalue=1, offvalue=0)

    '''
    en1 = Entry(root, textvariable=pageList, width=50,)
    #ch1.pack()
    pageList.set('商标信息')
    en1.grid(row=3, column=1, pady=10, padx=40)
    '''
    ch1 = tkinter.Checkbutton(root, text='商标', variable=var1, onvalue=1, offvalue=0)
    ch1.grid(row=3, column=1, stick=W, pady=10, padx=5)

    ch2 = tkinter.Checkbutton(root, text='经销企业', variable=var2, onvalue=1, offvalue=0)
    ch2.grid(row=4, column=1, stick=W, pady=10, padx=5)

    ch3 = tkinter.Checkbutton(root, text='不适宜人群', variable=var3, onvalue=1, offvalue=0)
    ch3.grid(row=5, column=1, stick=W, pady=10, padx=5)

    ch4 = tkinter.Checkbutton(root, text='许可证编号', variable=var4, onvalue=1, offvalue=0)
    ch4.grid(row=6, column=1, stick=W, pady=10, padx=5)

    ch5 = tkinter.Checkbutton(root, text='执行标准', variable=var5, onvalue=1, offvalue=0)
    ch5.grid(row=7, column=1, stick=W, pady=10, padx=5)


    btn3 = Button(root, text="识别 ", width=20,  bg='SkyBlue',command=Recogpdf)
    btn3.grid(row=8, column=0, stick=W,  pady=10, padx=50)

    lb3 = Label(root, text='', width=50)
    lb3.grid(row=8, column=1)

    llbe = Label(root, text='请输入要拆分出来的页码', width=20)
    llbe.grid(row=9, column=0, stick=W,pady=10,padx=50)

    een1 = Entry(root, textvariable=pageList, width=50,)
    pageList.set('例如:2表示将1,2拆出来，不填则1页拆分')
    een1.grid(row=9, column=1, pady=10, padx=40)

    bbtn3 = Button(root, text="拆 分 ", width=20,  bg='SkyBlue',command=splitPdf)
    bbtn3.grid(row=10, column=0, stick=W,  pady=10, padx=50)

    llb3 = Label(root, text='', width=50)
    llb3.grid(row=10, column=1)


    root.mainloop()