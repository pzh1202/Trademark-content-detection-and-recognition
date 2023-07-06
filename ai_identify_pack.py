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
        cursor = mongo_db.settings.find().sort('gt', -1).limit(1)
        if cursor.count() > 0:
            rec = list(cursor)[0]
            setting = rec["setting"]
        print(setting["log_output"])

        #define const
        output_folder = "/pack/output"
        img_output_folder = "/pack/temp/jpg"

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

            for img in tqdm(get_file_list(img_output_folder, p_postfix=['.jpg'])):
                file_name = os.path.basename(img).split('.')[0]
                final_result = recog_label(recognition, img, output_folder, file_name, logo_file)
                final_results.append(final_result)
                os.remove(img)
                print(file_name)


            #update collection tasks
            tasks.update_one({"taskid": taskid}, {"$set":{"finished": True}})
            tasks.update_one({"taskid": taskid}, {"$set": {"finishtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}})

            #insert one task's result
            insert_results(results, taskid, final_results)



        time.sleep(setting["task_interval"])
