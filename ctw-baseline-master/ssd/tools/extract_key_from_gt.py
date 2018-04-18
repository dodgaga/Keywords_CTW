# -*- coding: UTF-8 -*-
import analyse_anno
import json
from collections import defaultdict
import codecs
import re
import numpy as np

keyword1 = u'^\u9152'
keyword2 = u'^\u5e97'

def point_dist_to_line(p1,p2,p3):
    #computer the distance from p3 to p1 - p2
    return np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2 - p1)
    
def keywords_line(target_path):    
    
    lines = analyse_anno.readLines()
   
    max_mean = 0
    max_var = 0
    means = []
    varss = []
    for line in lines:
        anno = json.loads(line.strip())
        fp = codecs.open(target_path,"a+","utf-8")
        for block in anno['annotations']:
            counts = defaultdict(int)
            sentence = []
            x_center = []
            y_center = []
            flag = False
            pre_flag = False
            for char in block:
                if char['is_chinese']:
                    box = char['adjusted_bbox']
                    text = char['text']
                    #print text
                    sentence.append(text)
                    x_center.append(box[0]+ box[2]/2)
                    y_center.append(box[1] + box[3]/2)
                    counts[text] += 1
                    #zhongguo
                    if(pre_flag and re.match(keyword2, text)):
                        pre_flag = False
                        flag = True
                        p2_x = box[0]+ box[2]/2
                        p2_y = box[1] + box[3]/2
                    else:
                        pre_flag = False
                        
                    if(re.match(keyword1, text)):
                        pre_flag = True
                        p1_x = box[0]+ box[2]/2
                        p1_y = box[1] + box[3]/2
                  
                        
                        #print(u"^\u4e2d")
                        #print("\u56fd")
            if (flag):
                p1 = np.array([p1_x, p1_y])
                p2 = np.array([p2_x, p2_y])
                p3 = np.stack((x_center,y_center),axis=-1)
                result = []
                for p in p3:
                    result.append(point_dist_to_line(p1,p2,p))
                m = np.mean(np.array(result))
                v = np.var(np.array(result))
                means.append(np.mean(np.array(result)))
                varss.append(np.var(np.array(result)))
                #max_mean = x for x in means if x > max_mean else max_mean
                
                if m > max_mean:
                    max_mean = m
               
                if v > max_var:
                    max_var = v
                    
                    
                print(anno['file_name'])
                fp.write(anno['file_name']+ " ")
                for c in sentence:            
                    fp.write(c)
               # fp.write(" means:"+str(means)+ " ")
               # fp.write(" vars:"+ str(vars) + " ")
                fp.write("\n")
        fp.close()
        
    print("max_var",str(max_var))
    print(("max_mean",str(max_mean)))
    return means,varss
    

         

if __name__ == "__main__":
    source_path = []
    source_path = ""
    target_path = "gt_chengshi"
    keywords_line(target_path) 
