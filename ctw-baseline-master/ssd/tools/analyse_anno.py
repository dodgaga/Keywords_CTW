import json
import os
#import settings
from scipy import optimize  
from collections import defaultdict
import re
import numpy as np

#from pythonapi import anno_tools
import codecs

def f_1(x, A, B):
    return A*x + B  
    
def point_dist_to_line(p1,p2,p3):
    #computer the distance from p3 to p1 - p2
    return np.linalg.norm(np.corss(p2-p1, p1-p3)) / np.linalg.norm(p2 - p1)

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
                    dis_ratio = (x_begin - x_pre_end)/max((pre_width+box[2])/2,
                    (pre_height+box[3])/2)
                    distance.append(dis_ratio)
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
    

def analyse_key_data(lines):    
    k = []
    width_pair = []
    height_pair = []
    distance = []
    distance_ratio = []
    distance_no_abs = []
    board = []
    for line in lines:
        anno = json.loads(line.strip())
        #fp = codecs.open("temp.txt","a+","utf-8")        
        for block in anno['annotations']:
            x_center = []
            y_center = []
            flag = True
            x_pre_end = 0
            x_begin = 0
            pre_width = 0
            pre_height = 0
            pre_flag = False
            key_flag = False
            pre_cor = []
            cor = []
            counts = defaultdict(int)
            for char in block:
                box = char['adjusted_bbox']
                if char['is_chinese']:
                    text = char['text']
                    #print text
                    
                    counts[text] += 1
                    #zhongguo
                    if(pre_flag and re.match(u'^\u56fd', text)):
                        pre_flag = False
                        key_flag = True
                        
                        cor.append(box[0]+box[2]/2)
                        cor.append(box[1]+box[3]/2)
                        try:
                            dis = np.sqrt(np.sum((np.array(pre_cor) - np.array(cor))**2))
                            dis_no_abs = cor[0] - pre_cor[0] 
                            if(dis_no_abs<0):
                                print("image is left-right-reverse %s" %anno['file_name'])
                            
                        except:
                            print("pre_cor",len(pre_cor))
                            print("cor",len(cor))
                        
                        #x_begin = box[0]
                        #distance
                        #print dis
                        distance_no_abs.append(dis_no_abs/((pre_width+box[2])/2))
                        distance.append(dis)
                        board.append(max((pre_width+box[2])/2,
                    (pre_height+box[3])/2))
                    
                        dis_ratio = dis/max((pre_width+box[2])/2,
                    (pre_height+box[3])/2)
                        distance_ratio.append(dis_ratio)
                        cor = []
                        #print("*************")
                        if (dis_ratio > 3.42):
                            print("image is dis_taito too big %s" %anno['file_name'])
                            #print("the word is %s" %char['text'])
                        #x_pre_end = box[0]+box[2]
                        #width and height
                        width_pair.append(box[2]/pre_width)
                        height_pair.append(box[3]/pre_height)
                        #pre_width = box[2]
                        #pre_height = box[3]
                        
                    else:
                        pre_flag = False
                        pre_cor = []
                        x_pre_end = []
                        pre_width = []
                        pre_height = []
                        
                    if(re.match(u'^\u4e2d', text)):
                        pre_flag = True
                        x_pre_end = box[0]+box[2]
                        pre_width = box[2]
                        pre_height = box[3]
                        pre_cor.append(box[0]+box[2]/2)
                        pre_cor.append(box[1]+box[3]/2)
                
                
            
        #fp.close()
    return width_pair, height_pair, distance, board, distance_ratio,distance_no_abs    
     

if __name__ == "__main__":
    lines = readLines()
   # print("*************")
    width_pair, height_pair, distance, board, distance_ratio = analyse_key_data(lines)
    
    '''
    with open('distance_board','w+') as f:
        for i,j in enumerate(board):
            f.write(str(board[i]) +',' + str(distance[i]) + '\n')
    f.close()
    '''
