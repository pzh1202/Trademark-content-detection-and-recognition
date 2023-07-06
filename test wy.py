from tkinter import *

from tkinter import *
from tkinter.messagebox import *

from PdfMake import PDFMake

from tkinter import *
import tkinter.filedialog
from tkinter import *
import tkinter.filedialog
from PdfMake import PDFMake
from predict import recog_label

from wind import *

dict_result = {'商标':'false', '经销企业':'false', '不适宜人群':'false', '许可证编号':'false', '执行标准':'false', '条形码':'false', '保健品商标':'false', '成分':'false', '功效':'false', 'r标':'false'}


class LoginPage(object):
    def __init__(self, master=None):
        self.root = master  # 定义内部变量root
        self.root.geometry('%dx%d' % (300, 180))  # 设置窗口大小
        self.username = StringVar()
        self.password = StringVar()
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack()
        Label(self.page).grid(row=0, stick=W)
        Label(self.page, text='账户: ').grid(row=1, stick=W, pady=10)
        Entry(self.page, textvariable=self.username).grid(row=1, column=1, stick=E)
        Label(self.page, text='密码: ').grid(row=2, stick=W, pady=10)
        Entry(self.page, textvariable=self.password, show='*').grid(row=2, column=1, stick=E)
        Button(self.page, text='登陆', command=self.loginCheck).grid(row=3, stick=W, pady=10)
        Button(self.page, text='退出', command=self.page.quit).grid(row=3, column=1, stick=E)

    def loginCheck(self):
        name = self.username.get()
        secret = self.password.get()
        self.page.destroy()
        MainPage(self.root)
        '''
        if name == 'wangliang' and secret == '123456':
            self.page.destroy()
            MainPage(self.root)
        else:
            showinfo(title='错误', message='账号或密码错误！')
        '''

class MainPage(object):
    def __init__(self, master=None):
        self.root = master  # 定义内部变量root
        self.root.geometry('%dx%d' % (600, 400))  # 设置窗口大小
        self.createPage()

    def createPage(self):
        self.inputPage = InputFrame(self.root)  # 创建不同Frame
        self.queryPage = QueryFrame(self.root)
        self.countPage = CountFrame(self.root)

        self.inputPage.pack()  # 默认显示数据录入界面
        menubar = Menu(self.root)
        menubar.add_command(label='识别', command=self.inputData)
        menubar.add_command(label='分割', command=self.queryData)
        menubar.add_command(label='功效增减', command=self.countData)

        self.root['menu'] = menubar  # 设置菜单栏

    def inputData(self):
        self.inputPage.pack()
        self.queryPage.pack_forget()
        self.countPage.pack_forget()


    def queryData(self):
        self.inputPage.pack_forget()
        self.queryPage.pack()
        self.countPage.pack_forget()

    def countData(self):
        self.inputPage.pack_forget()
        self.queryPage.pack_forget()
        self.countPage.pack()


    def aboutDisp(self):
        self.inputPage.pack_forget()
        self.queryPage.pack_forget()
        self.countPage.pack_forget()
        #self.aboutPage.pack()

class InputFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.importPrice = StringVar()
        self.filename = StringVar()
        self.sellPrice = StringVar()
        self.deductPrice = StringVar()
        self.outputdirectory = StringVar()
        self.pdfFileName = StringVar()
        self.pdfFileNames = StringVar()
        self.pageList = StringVar()
        self.var1 = BooleanVar()
        self.var2 = BooleanVar()
        self.var3 = BooleanVar()
        self.var4 = BooleanVar()
        self.var5 = BooleanVar()
        self.var6 = BooleanVar()
        self.var7 = BooleanVar()
        self.var8 = BooleanVar()
        self.var9 = BooleanVar()
        self.var10 = BooleanVar()
        self.flag = 0
        self.createPage()


    def createPage(self):
        Label(self).grid(row=0, stick=W, pady=10)
        lb0 = Label(self, text='识别检测', width=20, font=("华文细黑", 18), bg='black', fg="white")
        lb0.grid(row=0, columnspan=2, stick=W, pady=10, )
        btn1 = Button(self, text="选择需要识别的文件", width=20, command=self.getPdfFileName)
        btn1.grid(row=1, column=0, stick=W, pady=10, padx=50)
        global lb1
        lb1 = Label(self, text='', width=40, bg='gray')
        lb1.grid(row=1, column=1)
        btn2 = Button(self, text="识别后的存储路径", width=20, command=self.selectPath)
        btn2.grid(row=2, column=0, stick=W, pady=10, padx=50)
        global lb2
        lb2 = Label(self, text='', width=40, bg='gray')
        lb2.grid(row=2, column=1)
        lbe = Label(self, text='需要识别的内容', width=20)
        lbe.grid(row=3, column=0, stick=W, pady=10, padx=50)
        ch1 = tkinter.Checkbutton(self, text='商标', variable=self.var1, onvalue=1, offvalue=0)
        ch1.grid(row=3, column=1, stick=W, pady=10, padx=5)
        ch2 = tkinter.Checkbutton(self, text='经销企业', variable=self.var2, onvalue=1, offvalue=0)
        ch2.place(x=350, y=160)
        ch3 = tkinter.Checkbutton(self, text='不适宜人群', variable=self.var3, onvalue=1, offvalue=0)
        ch3.place(x=450, y=160)
        ch5 = tkinter.Checkbutton(self, text='执行标准', variable=self.var5, onvalue=1, offvalue=0)
        ch5.grid(row=4, column=1, stick=W, pady=10, padx=5)
        ch6 = tkinter.Checkbutton(self, text='条形码', variable=self.var6, onvalue=1, offvalue=0)
        ch6.place(x=350, y=207)
        ch7 = tkinter.Checkbutton(self, text='保健品商标', variable=self.var7, onvalue=1, offvalue=0)
        ch7.place(x=450, y=207)
        ch4 = tkinter.Checkbutton(self, text='许可证编号', variable=self.var4, onvalue=1, offvalue=0)
        ch4.grid(row=5, column=1, stick=W, pady=10, padx=5)
        ch8 = tkinter.Checkbutton(self, text='成分表', variable=self.var8, onvalue=1, offvalue=0)
        ch8.place(x=350, y=254)
        ch9 = tkinter.Checkbutton(self, text='功效', variable=self.var9, onvalue=1, offvalue=0)
        ch9.place(x=450, y=254)
        ch10 = tkinter.Checkbutton(self, text='r标', variable=self.var10, onvalue=1, offvalue=0)
        ch10.grid(row=6, column=1, stick=W, pady=10, padx=5)

        btn3 = Button(self, text="识别 ", width=20, bg='SkyBlue', command=self.Recogpdf)
        btn3.grid(row=7, column=0, stick=W, pady=10, padx=50)
        global lb3
        lb3 = Label(self, text='', width=40, bg='gray')
        lb3.grid(row=7, column=1)

    def getPdfFileName(self):
        self.filename = tkinter.filedialog.askopenfilename()
        self.pdfFileName.set(self.filename)
        print(self.filename)

        if self.filename != '':
            lb1.config(text="识别文件：" + self.filename)
        else:
            lb1.config(text="您没有选择任何文件")

    def selectPath(self):
        path = tkinter.filedialog.askdirectory()
        self.outputdirectory.set(path)
        if path != '':
            lb2.config(text="输出路径：" + path)
        else:
            lb2.config(text="您没有选择任何路径")

    def Recogpdf(self):
        if self.outputdirectory.get() != '' and self.pdfFileName.get() != '':
            if True:
                # master.spilt_pdf(pdfFileName.get())
                for k, m in zip((self.var1, self.var2, self.var3, self.var4, self.var5, self.var6, self.var7, self.var8, self.var9, self.var10),
                                ('商标', '经销企业', '不适宜人群', '许可证编号', '执行标准', '条形码', '保健品商标', '成分', '功效', 'r标')):
                    dict_result[m] = (k.get() == 1)
                print(dict_result)

                recog_label(self.pdfFileName.get(), self.outputdirectory.get(), dict_result)
                lb3.config(text="是个可靠的正品！")

                tk = Toplevel()
                tk.title('检测图片')
                #photo = PhotoImage(file="12.png")
                photo = PhotoImage(file="merged.png")
                theLabel = Label(tk, image=photo)
                label = Label(tk, text="正品", font=('heiti', 20), fg='red', bg='pink')
                label.place(x=10, y=10)
                theLabel.pack()
                tk.mainloop()

        else:
            lb3.config(text="选择文件或者目录有误！")


class QueryFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.outputdirectory = StringVar()
        self.pdfFileName = StringVar()
        self.pdfFileNames = StringVar()
        self.pageList = StringVar()
        self.createPage()


    def createPage(self):
        Label(self).grid(row=0, stick=W, pady=10)
        lb0 = Label(self, text='文件拆分拆解', width=20, font=("华文细黑", 18), bg='black', fg="white")
        lb0.grid(row=0, columnspan=3, stick=W, pady=10, )
        btn1 = Button(self, text="选择需要分割的文件", width=20, command=self.getPdfFileName)
        btn1.grid(row=1, column=0, stick=W, pady=10, padx=50)
        global llb1
        llb1 = Label(self, text='', width=40, bg='gray')
        llb1.grid(row=1, column=1)
        btn2 = Button(self, text="分割后的存储路径", width=20, command=self.selectPath)
        btn2.grid(row=2, column=0, stick=W, pady=10, padx=50)
        global llb2
        llb2 = Label(self, text='', width=40, bg='gray')
        llb2.grid(row=2, column=1)

        llbe = Label(self, text='请输入要拆分出来的页码', width=20)
        llbe.grid(row=3, column=0, stick=W, pady=10, padx=50)

        een1 = Entry(self, textvariable=self.pageList, width=35)
        self.pageList.set('例如:2表示将1,2拆出来，不填则1页拆分')
        een1.grid(row=3, column=1, pady=10, padx=10)
        ''''''
        btn3 = Button(self, text="拆分", width=20, bg='SkyBlue', command=self.splitPdf)
        btn3.grid(row=4, column=0, stick=W, pady=10, padx=50)
        global llb3
        llb3 = Label(self, text='', width=40, bg='gray')
        llb3.grid(row=4, column=1)

    def getPdfFileName(self):
        self.filename = tkinter.filedialog.askopenfilename()
        self.pdfFileName.set(self.filename)
        print(self.filename)

        if self.filename != '':
            llb1.config(text="识别文件：" + self.filename)
        else:
            llb1.config(text="您没有选择任何文件")

    def selectPath(self):
        path = tkinter.filedialog.askdirectory()
        self.outputdirectory.set(path)
        if path != '':
            llb2.config(text="输出路径：" + path)
        else:
            llb2.config(text="您没有选择任何路径")

    def splitPdf(self):
        if self.outputdirectory.get() != '' and self.pdfFileName.get() != '':
            master = PDFMake(outputdirectory=self.outputdirectory.get())
            if self.pageList.get() == '':
                master.spilt_pdf(self.pdfFileName.get())
                llb3.config(text="拆分成功！")
            else:
                try:
                    master.spilt_pdf(self.pdfFileName.get(), self.pageList.get())
                    llb3.config(text="拆分成功！")
                except ValueError:
                    llb3.config(text="拆分失败:ValueError！")
        else:
            llb3.config(text="拆分失败:选择文件或者目录有误！")



class CountFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.word = StringVar()
        self.createPage()

    def createPage(self):
        Label(self).grid(row=0, stick=W, pady=10)
        lb0 = Label(self, text='功效词汇增减', width=20, font=("华文细黑", 18), bg='black', fg="white")
        lb0.grid(row=0, columnspan=3, stick=W, pady=10, )
        llbe = Label(self, text='请输入要增加或删除的词汇', width=20)
        llbe.grid(row=1, column=0, stick=W, pady=10, padx=50)

        een1 = Entry(self, textvariable=self.word, width=35)
        self.word.set('请逐词填写，例如：美容')
        een1.grid(row=1, column=1, pady=10, padx=10)

        bbbtn3 = Button(self, text="增加", width=20, bg='SkyBlue', command=self.add_word)
        bbbtn3.grid(row=2, column=0, stick=W, pady=10, padx=50)

        bbbtn4 = Button(self, text="删除", width=20, bg='SkyBlue', command=self.delete_word)
        bbbtn4.grid(row=2, column=1, stick=W, pady=10, padx=50)

        bbbtn5 = Button(self, text="展示", width=20, bg='SkyBlue', command=self.tes)
        bbbtn5.grid(row=3, column=0, stick=W, pady=10, padx=50)

        global text_pre
        scroll = tkinter.Scrollbar()
        text_pre = tkinter.Text(self, width=30, height=10)
        scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        scroll.config(command=text_pre.yview)
        text_pre.config(yscrollcommand=scroll.set)
        text_pre.grid(row=3, column=1, stick=W, pady=10, padx=50)

    def add_word(self):
        flag = 0
        f2 = open('word.txt')
        source = f2.read()
        if len(re.findall(self.word.get(), source)) > 0:
            f2.close()
            showinfo(title='错误', message='词条已存在')
            flag = 1
        if flag == 0:
            info = self.word.get() + '\n'
            with open('word.txt', 'a') as f3:
                f3.write(info)
            f3.close()
            #f.flush()

    def delete_word(self):
        # , encoding="utf-8"
        f2 = open('word.txt')
        source = f2.read()
        if len(re.findall(self.word.get(), source)) == 0:
            f2.close()
            showinfo(title='错误', message='词条不存在')
        lines = (i for i in open('word.txt', 'r') if self.word.get() not in i and i != '\n')
        f = open('word_new.txt', 'w')
        f.writelines(lines)
        f.close()
        os.rename('word.txt', 'word_tmp.txt')
        os.rename('word_new.txt', 'word.txt')
        os.remove('word_tmp.txt')



    def tes(self):
        f1 = open('word.txt')
        result = list()
        for line in f1.readlines():  #
            line = line.strip()  #
            if not len(line):  #
                continue
            result.append(line)
        print(result)
        f1.close()
        text_pre.delete('1.0','end')
        text_pre.insert(INSERT, result)



#123
class AboutFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.createPage()

    def createPage(self):
        Label(self, text='关于界面').pack()

if __name__ == '__main__':
    dict_result = {'商标': 'false', '经销企业': 'false', '不适宜人群': 'false', '许可证编号': 'false', '执行标准': 'false', '条形码': 'false',
                   '保健品商标': 'false', '成分': 'false', '功效': 'false', 'r标':'false'}

    root = Tk()
    root.title('南京同仁堂')
    LoginPage(root)
    root.mainloop()
