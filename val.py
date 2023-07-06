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

def init_args():
    import  argparse
    parser = argparse.ArgumentParser(description='DetNJTRT.pytorch')
    parser.add_argument('--input_folder', default='./pdf/保健食品', type=str, help='pdf for detect')
    parser.add_argument('--output_folder', default='./pdf/picture',type=str, help='img path for output')
    #parser.add_argument('--save_result', action='store_true', help='save result to txt file')
    args = parser.parse_args()
    return args

def get_file_list(folder_path: str, p_postfix: list = None, sub: bool = True) -> list:
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.pdf']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import pymongo
    client = pymongo.MongoClient(host='localhost')
    final_results = []
    db = client.test
    collection = db.njtrt
    args = init_args()
    print(args)
    pdf_folder = pathlib.Path(args.input_folder)
    img_output_folder = '/home/trt/trt_pro/pdf/image'
    dict_result = {'商标': 'True', '经销企业': 'True', '不适宜人群': 'True', '许可证编号': 'True', '执行标准': 'True', '条形码': 'True',
                   '保健品商标': 'True', '成分': 'True', '功效': 'True', 'r标': 'True'}


    for pdf in tqdm(get_file_list(args.input_folder, p_postfix=['.pdf'])):
        file_name = os.path.basename(pdf).split('.')[0]
        print(file_name)
        master = PDFMake(outputdirectory=img_output_folder)
        master.spilt_pdf(pdf, 1, name = file_name)


    for img in tqdm(get_file_list(img_output_folder, p_postfix=['.jpg'])):
        file_name = os.path.basename(img).split('.')[0]
        final_result = recog_label(img, args.output_folder, dict_detect, file_name)
        final_results.append(final_result)
        print(file_name)

    result = []
    for i in final_results:
        #print({'name': i['name']})
        res = collection.delete_many({'name': i['name']})
        #print(res.deleted_count)
        result.append(collection.insert_one(i))

    #print(result)
