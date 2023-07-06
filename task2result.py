import os
import sys
import glob
from natsort import natsorted
from PdfMake import PDFMake#
from predict import recog_label#
from tqdm import tqdm
import pymongo
import requests
import copy
import datetime

# 连接数据库
database_name = 'trt'
host = 'localhost'


#数据库信息存放
dict_detect = {}

url_title = 'packUrl'
format_title = 'packFomat'
id_title = 'Id'
state_title = 'isFinished'


dir_seg = './dir_segmentation/'
dir_rec = './dir_recognition/'
dir_input = './dir_input/'

info_flag = True #是否输出info日志
error_flag = True#是否输出error日志
error_exit_flag = False#遇到error是否中断
remove_flag = True#中间文件是否用完就删

value_title = 'Value'
name_title = 'Name'

# 输出日志，taskid，level=true 没毛病，level=false有错误， msg要输出什么信息，collection输出到哪个集合    
#\033[1;31;40m    <!--1-高亮显示 31-前景色红色  40-背景色黑色-->
#\033[0m          <!--采用终端默认设置，即取消颜色设置-->
def log_output(taskid, level, msg, collection):
    gt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if level == True:
        level = 'Info'
        print('\033[4m'+msg+'\033[0m')
        if info_flag != False:
            collection.insert_one({'gt': gt, 'taskid': taskid, 'msg':msg, 'level':level})
    else:
        level = 'Error'
        print('\033[1;31;40m'+msg+'\033[0m')
        if error_flag != False:
            collection.insert_one({'gt': gt, 'taskid': taskid, 'msg':msg, 'level':level})
        if error_exit_flag: 
            sys.exit()



def get_id_list():
    args = []
    try:
        assert len(sys.argv) > 1 , "taskid not found"
    except:
        log_output(0, False, 'ARGV ERROR: arg not found',logs_c)
    else:
        try:
            args = [int(arg) for arg in sys.argv[1:]]
        except:
            log_output(0, False, 'ARGV ERROR: argv[?] is not a int?', logs_c)
        else:
            log_output(0, True, 'argv info: '+str(args), logs_c)
    return args


def get_collections(db):
    logs_c = db.logs
    results_c = db.results
    settings_c = db.settings
    tasks_c = db.tasks
    detect_c = db.detect
    return logs_c, results_c, settings_c, tasks_c, detect_c

def read_document(collection, var, name):
    flag = False
    for task in collection.find({'Name': name}):
        if flag == True:
            log_output(0,False,'SETTINGS ERROR: ' + name + ' is repeated', logs_c)
        flag = True
        update_var = task[value_title]
        print(name+'(updated)'+':', update_var )
        #return update_var

    if flag == False:
        update_var = var
        log_output(0,False,'SETTINGS ERROR: ' + name+' not found', logs_c)
        print(name+'(default)'+ ':', update_var )

    # 返回默认值
    return update_var

def read_settings(collection):
    global info_flag, error_flag, error_exit_flag, remove_flag, url_title, format_title, state_title, id_title, dir_seg, dir_rec, dir_input
    info_flag = read_document(collection, info_flag, 'info_flag')
    error_flag = read_document(collection, error_flag, 'error_flag')
    error_exit_flag = read_document(collection, error_exit_flag, 'error_exit_flag')
    remove_flag = read_document(collection, remove_flag, 'remove_flag')
    url_title = read_document(collection, url_title, 'url_title')
    format_title = read_document(collection, format_title, 'format_title')
    state_title = read_document(collection, state_title, 'state_title')
    id_title = read_document(collection, id_title, 'id_title')
    dir_seg = read_document(collection, dir_seg, 'dir_seg')
    dir_rec = read_document(collection, dir_rec, 'dir_rec')
    dir_input = read_document(collection, dir_input, 'dir_input')

def connect_to_db(host, database_name):
    client = pymongo.MongoClient(host=host)
    db = client[database_name]
    print('--- Database Message ---')
    print('client_list: ', client.list_database_names())
    print('db message: ', db)
    print('*'*50)
    return client, db

# 获取文件列表
def get_file_list(folder_path: str, p_postfix: list = None, sub: bool = True) -> list:
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.pdf']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)

# 根据url下载
def download_from_url(url, name):
    print("---downloading with requests---")
    url = 'http://gateway.ntdjk.com/blade-resource/oss/endpoint/file-view-link?attachId=1420763982380843009'
    #url = 'http://www.baidu.com'
    r = requests.get(url)
    with open(name+".pdf", "wb") as code:
        code.write(r.content)
        
## 根据task下载
def download_from_task(task, output_dir):
    download_url = task[url_title]
    pdf_name = str(task[id_title])
    # 创建专属目录 dir_input/pdf_name(Id)/
    special_dir = output_dir + '/' + pdf_name + '/'
    if not os.path.exists(special_dir):
        os.makedirs(special_dir)
    download_from_url(download_url, special_dir+pdf_name)
    return special_dir

## 分割
def segmentation_pdf(input_dir, output_dir):
    print("---segmenting---")
    assert len(get_file_list(input_dir, p_postfix=['.pdf'])) != 0, 'no pdf found in'+input_dir

    for pdf in tqdm(get_file_list(input_dir, p_postfix=['.pdf'])):
        file_name = os.path.basename(pdf).split('.')[0]
        special_dir = output_dir+'/'+file_name +'/'
        if not os.path.exists(special_dir):
            os.makedirs(special_dir)
        master = PDFMake(outputdirectory=special_dir)
        master.spilt_pdf(pdf, 1, name = file_name)
        #print('segmentation file name:',file_name)
    return output_dir+file_name
        
        
## 识别
def recognition_img(input_dir, output_dir):
    print('---recognizing---')
    assert len(get_file_list(input_dir, p_postfix=['.jpg'])) != 0, 'no img found in '+input_dir
    for img in tqdm(get_file_list(input_dir, p_postfix=['.jpg'])):
        file_name = os.path.basename(img).split('.')[0]
        special_dir = output_dir+'/'+file_name +'/'
        if not os.path.exists(special_dir):
            os.makedirs(special_dir)
        final_result = recog_label(img, special_dir, dict_detect, file_name)
        return final_result
        
# 重组字典，写入result
def insert_result(final_result, task, to_collection):
    print('---inserting---')
    final_one = {}
    for index in final_result:
        if index == 'name':
            pass
        else:
            final_one['taskId'] = task[id_title]
            final_one['auditName'] = index
            final_one['scanFount'] = final_result[index][0]
            final_one['scanContent'] = final_result[index][1]
            print('---final_one---')
            print(final_one)
            #=========================
            temp = copy.deepcopy(final_one)
            result = to_collection.insert_one(temp)
            
            
# 更新状态
def task_is_finished(Id, collection, state = True):
    print('---updating---')
    old_value_id = {id_title: Id}
    new_value = {"$set": {state_title: state}}

    collection.update_one(old_value_id, new_value)


# if 'task' do not have a packUrl?, not a pdf?
def operation_id(tasks_collection, id_number):
    print('===== processing', id_number, '=====')
    flag = False
    for task in tasks_collection.find({id_title: id_number}):
        # 判断id是否重复
        if flag == True:
            log_output(task[id_title],False,'ID ERROR: taskid'+str(id_number)+ ' is repeated',logs_c)
            break
        flag = True
        if task[state_title] == True:
            log_output(task[id_title],False,'TASK ERROR: task is already finished',logs_c)
            #break
        if task[format_title] != 'pdf':
            log_output(task[id_title],False,'FILE ERROR: not a PDF file',logs_c)
            break
        #time1 = datetime.datetime.now()
        download_url = task[url_title]

        # 下载
        try:
            special_dir = download_from_task(task, dir_input)
        except:
            information = repr(sys.exc_info()[1])
            log_output(task[id_title], False, 'DOWNLOAD ERROR: '+ information, logs_c)
        else:
            log_output(task[id_title], True, 'download info: '+special_dir , logs_c)

        # 分割
        try:
            special_dir = segmentation_pdf(special_dir, dir_seg)
        except:
            #information = sys.exc_info()[1].args[0]
            information = repr(sys.exc_info()[1])
            log_output(task[id_title], False, 'SEGMENTATION ERROR: ' + information, logs_c)
        else:
            log_output(task[id_title], True, 'segmentation complete', logs_c)


        # 识别
        try:
            final_result = recognition_img(special_dir, dir_rec)
        except:
            information = repr(sys.exc_info()[1])
            log_output(task[id_title], False, 'RECOGNITION ERROR: ' + information, logs_c)
        else:
            log_output(task[id_title], True, 'recognition complete', logs_c)


        # 插入results
        try:
            insert_result(final_result, task, results_c)
        except:
            information = repr(sys.exc_info()[1])
            log_output(task[id_title], False, 'INSERT ERROR: ' + information, logs_c)
        else:
            log_output(task[id_title], True, 'insert to result succeeded', logs_c)

        # 更新状态
        try:
            task_is_finished(task[id_title], collection=tasks_collection, state = True)
        except:
            information = repr(sys.exc_info()[1])
            log_output(task[id_title], False, 'UPDATE ERROR: ' + information, logs_c)
        else:
            log_output(task[id_title], True, 'state info: state update succeeded', logs_c)

        #break
    if flag == False:
        log_output(id_number, False,'ID ERROR: '+str(id_number) +' not found',logs_c)

#从数据库中读取需要检测的内容，存入字典数据dict_detect中
def readDetect(detect_collection):
    for tasks in detect_collection.find():
        #key_list = tasks.keys()
        del[tasks['_id']]
        dict_detect.update(tasks)
            
#operation_from_collection()
client, db = connect_to_db(host, database_name)
logs_c, results_c, settings_c, tasks_c, detect_c = get_collections(db)
readDetect(detect_c)

if __name__ == '__main__':
    try:
        client, db = connect_to_db(host, database_name)
        logs_c, results_c, settings_c, tasks_c, detect_c = get_collections(db)

    except:
        log_output(0,False, 'DB ERROR: CONNECTION FAILED', logs_c)
    else:
        log_output(0,True, 'db info: connect sucessfully', logs_c)


#将数据库中需要检测项读入dict_detect字典中
    read_settings(settings_c)

    id_list = get_id_list()
    for id_number in id_list:
        operation_id(tasks_c, id_number)

    os.system('rm -r '+ dir_input)
    os.system('rm -r '+ dir_rec)
    os.system('rm -r '+ dir_seg)
    #ipdb.set_trace()
    #operation_id(tasks_c, 123456)
    #ipdb.set_trace()
    #def log_output(taskid, level, msg, collection):
        #time2 = datetime.datetime.now()
