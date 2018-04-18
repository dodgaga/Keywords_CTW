import json
import os
#import settings
from scipy import optimize  

from pythonapi import anno_tools
import codecs

def f_1(x, A, B):
    return A*x + B  

def readLines():    
    with open('/home/wudao/ctw/ctw-annotations/train.jsonl') as f:
        lines = f.read().splitlines()
    with open('/home/wudao/ctw/ctw-annotations/val.jsonl') as f:
        lines += f.read().splitlines()
    return lines

def analyse_data(lines):    
    k = []
    width_pair = []
    height_pair = []
    distance = []
    for line in lines:
        anno = json.loads(line.strip())
        fp = codecs.open("temp.txt","a+","utf-8")        
        for block in anno['annotations']:
            x_center = []
            y_center = []
            flag = True
            x_pre_end = 0
            x_begin = 0
            pre_width = 0
            pre_height = 0
            
            for char in block:
                box = char['adjusted_bbox']
                if(flag):
                    x_pre_end = box[0]+box[2]
                    pre_width = box[2]
                    pre_height = box[3]
                    flag = False
                else:
                    x_begin = box[0]
                    distance.append(x_begin - x_pre_end)
                    #print("*************")
                    if(x_begin - x_pre_end > 200):
                        
                        fp.write(anno['file_name']+ " "+ char['text'] + "\n")
                        print("image is %s" %anno['file_name'])
                        print("the word is %s" %char['text'])
                    x_pre_end = box[0]+box[2]
                    
                    width_pair.append(box[2]/pre_width)
                    height_pair.append(box[3]/pre_height)
                    pre_width = box[2]
                    pre_height = box[3]
                    
                    

                x_center.append(box[0]+box[2]/2)
                y_center.append(box[1]+box[3]/2)
                
            if(len(x_center)>1):
                A1, B1 = optimize.curve_fit(f_1, x_center, y_center)[0]
                k.append(A1)
        fp.close()
    return k, width_pair, height_pair, distance

if __name__ == "__main__":
    lines = readLines()
   # print("*************")
    analyse_data(lines)
        
