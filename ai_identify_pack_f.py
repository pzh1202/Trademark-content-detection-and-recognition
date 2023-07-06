#import mongodb content
import configparser
import pymongo
import time

#import identify content
import os
import sys
import pathlib
import glob
import torch
import numpy
from natsort import natsorted
from PdfMake import PDFMake
from predict import recog_label
from task2result import dict_detect
import ai_identify_pack as aip
import pathlib
from tqdm import tqdm
from val import get_file_list
import traceback
import datetime



'''
#使用方法示例：

## 已有函数：   recog_label(recognition, img_path1, img_save_path, name, logo_file)

    recog_label_log = (log_decorator('recognition', logs, taskid,True,True,True))(recog_label)
    final_result = recog_label_log(recognition, img, output_folder, file_name, logo_file)

## recog_label过程中如果没有错误
    终端输出： 
    ```
    [info] - recognition
    ```
    同样的内容也会输出到 logs_collection

## 如果遇到错误
    终端输出： 
    ```
     [Error traceback]
     <$scriptname1>.py       :<$linenum1> 'ret = func(*args, **kargs)' in improved_function()
     <$scriptname2>.py       :<$linenum2> xxxxxx
     ......
     [Error traceback]
     [ERROR] - recognition: xxxxxx
    ```
    其中 `[ERROR] - recognition: xxxxxx` 会输出到 logs_collection

'''

def log_decorator(head_msg, collection, taskid=-1, info_flag=True, error_flag=True, error_exit_flag=True):
    def decorator(func):
        def improved_function(*args, **kargs):
            ret = None
            msg = 'no msg'
            level = False
            try:
                ret = func(*args, **kargs)
            except:
                level = False
                exc_type, exc_value, exc_tb = sys.exc_info()
                # 定位错误
                print( ' [Error traceback]' )
                for filename, linenum, funcname, source in traceback.extract_tb(exc_tb):
                    print(" %-23s:%s '%s' in %s()" % (filename, linenum, source, funcname))
                print( ' [Error traceback]' )
                msg = ' [ERROR] - ' + head_msg +':' + repr(exc_value)
            else:
                level = True
                msg = ' [info] - ' + head_msg
            log_output(level, msg, collection, taskid, info_flag, error_flag, error_exit_flag)
            return ret
        return improved_function
    return decorator




# echo msg，print and insert it to logs_collection
def log_output( level, msg, collection, taskid=-1, info_flag=True, error_flag=True, error_exit_flag=True):
    gt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if level == True:
        level = 'Info'
        #print('\033[4m'+' [info] - '+msg+'\033[0m')
        print('\033[1;32m'+msg+'\033[0m')#green
        if info_flag != False:
            collection.insert_one({'gt': gt, 'taskid': taskid, 'msg':msg, 'level':level})
    else:
        level = 'Error'
        #print('\033[1;31;40m'+' [Error] - '+msg+'\033[0m')
        print('\033[1;31;40m'+msg+'\033[0m')#red
        if error_flag != False:
            collection.insert_one({'gt': gt, 'taskid': taskid, 'msg':msg, 'level':level})
        if error_exit_flag == True: 
            sys.exit()




def get_collections(db):
    tasks = db["tasks"]
    results = db["results"]
    logs = db["logs"]
    settings = db["settings"]
    return tasks, results, logs, settings

def get_settings(collection):
    dict1 = list(collection.find().sort('gt', -1).limit(1))[0]
    setting = dict1["setting"]
    dict2 = list(collection.find().sort('gt', -1).limit(1))[0]
    recognition = dict2["recognition"]
    dict3 = list(collection.find().sort('gt', -1).limit(1))[0]
    logo_file = dict3["logo_file"]
    return setting, recognition, logo_file


def get_tasks(collection):
    cursor = collection.find({"finished":False}).limit(1)
    if cursor.count() > 0:
        mytask = list(collection.find({"finished":False}).limit(1))[0]
        taskid = mytask["taskid"]
        pack_source_files = mytask["pack_source_files"]
    else:
        taskid = -1
        pack_source_files = None
    return taskid, pack_source_files


def insert_results(collection, id, final_results):
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    collection.insert({"taskid":id,"result":final_results,"gt":datetime})


if __name__ == '__main__':
    while True:
        # read ini configuration file
        config = configparser.ConfigParser()
        config.read('conf.ini', encoding='GB18030')
        # connect to mongodb
        mongo_client = pymongo.MongoClient('mongodb://' + config.get('mongo', 'server') + ':' + config.get('mongo', 'port') + '/')
        mongo_db = mongo_client[config.get('mongo', 'database')]
        setting = {'task_interval': 0.433}
        cursor = mongo_db.settings.find().sort('gt', -1).limit(1)#按时间排序，取settings collection中最新的document
        if cursor.count() > 0:
            rec = list(cursor)[0]  # rec = settings collection中最新的document
            setting = rec["setting"] # rec中有很多个内容，比如setting(放配置信息)，recognition（放识别关键字）, logo_file（放logo）...
        print(setting["log_output"]) # ？

        #define const
        output_folder = "/pack/output"
        img_output_folder = "/pack/temp/jpg"
        text_source_folder = "/pack/temp/text"


        #define var
        final_results = []

        #get collection names
        tasks, results, logs, settings = get_collections(mongo_db)

        #get settings
        setting, recognition, logo_file = get_settings(settings)

        #get a task from mongo_db.tasks
        taskid, input_files = get_tasks(tasks)

        #execute the task been got
        if(taskid != -1):
            for pdf in input_files:
                file_name = os.path.basename(pdf).split('.')[0]
                print(file_name)
                master = PDFMake(outputdirectory=img_output_folder)
                master.spilt_pdf(pdf, 1, name=file_name)
                time.sleep(0.1)

            for img in tqdm(get_file_list(img_output_folder, p_postfix=['.jpg'])):
                file_name = os.path.basename(img).split('.')[0]
                recog_label_log = (log_decorator('recognition', logs, taskid, True, True, False))(recog_label)
                final_result = recog_label_log(recognition, img, text_source_folder, output_folder, file_name, logo_file)

                #final_result = recog_label(recognition, img, output_folder, file_name, logo_file)
                final_results.append(final_result)
                #delete temoorary files
                os.remove(img)
                for file in os.listdir(text_source_folder):
                    textfile = os.path.join(text_source_folder, file)
                    os.remove(textfile)
                print(file_name)

            #update collection tasks
            tasks.update_one({"taskid": taskid}, {"$set":{"finished": True}})
            tasks.update_one({"taskid": taskid}, {"$set": {"finishtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}})

            #insert one task's result
            insert_results(results, taskid, final_results)



        time.sleep(setting["task_interval"])
