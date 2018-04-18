#coding=utf-8
import numpy as np
import json
import pprint
from pythonapi import anno_tools
import codecs
import re
import sys
import LinkedList
from collections import defaultdict
from scipy import optimize 
import os
import cv2
import tools

keyword1 = u'^\u5bbe'
keyword2 = u'^\u9986'
DEBUG = 1
def f_1(x, A, B):
    return A*x + B  
    
def vis_detection(filepath, boxes, savepath):
    im = cv2.imread(filepath)
    #im = im[:,:,(2,1,0)].copy()
    im = im[:,:,].copy()

    for box in boxes:
        cv2.rectangle(im, (int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])),(0,0,255),3)
        
    cv2.imwrite(savepath,im)

def readLines():    
    with open('/home/wudao/ctw/ctw-annotations/train.jsonl') as f:
        lines = f.read().splitlines()
    with open('/home/wudao/ctw/ctw-annotations/val.jsonl') as f:
        lines += f.read().splitlines()
    return lines

def findIndex(L,e):
    return [i for (i,j) in enumerate(L) if j == e]
    
    #findKeywords
    #return: flag: has keyword?
    #         [], keyword's index pair
def findKeywords(image_boxes,image_texts):
    DEBUG = 1
    flag1 = flag2 = False
    location1 = [] 
    location2 = []
    
    index1 = []
    index2 = []
    key_pair = []
    
    for i, text in enumerate(image_texts):
        if re.match(keyword1, text):#zhong
            flag1 = True
            index1.append(i) 
        if re.match(keyword2, text):#guo
            flag2 = True
            index2.append(i)
            
    if DEBUG:        
        print("in the function")
        print("flag1",flag1)
        print("flag2",flag2)
        print ("index1", index1)
        print("index2", index2)
    
    if( not(flag1 and flag2)):
        return False, []
    
    #if has the char of keywords, from the distance and board ratio to judge whether they are words
    if len(index1) > len(index2):
        long = index1
        short = index2
    else:
        long = index2
        short = index1
    
    print short 
    
    location1 = [image_boxes[i] for i in short]
    location2 = [image_boxes[i] for i in long]
    
    pairs = {}
    
    for i, l1 in enumerate(location1):
        min_v = sys.maxint
        valid_index = 0
        is_valid = False
        for j, l2 in enumerate(location2):
            dis = tools.compute_distance_edge_ratio(l1,l2)
            #print("dis",dis)
            #if (dis<3.42 and dis > 0.169):
            edge_ratio = tools.compute_edge_ratio(l1,l2)
            #print("edge_ratio", edge_ratio)
            if(edge_ratio<7.78 and edge_ratio > 0.128):
                #print("edge_ratio", edge_ratio)
                #print("dis", dis)
                #if a char1 match n char2 or a char2 match n char1 select the min one(suppose value is tend to 1)
                d_var = abs(dis-1)
                e_var = abs(edge_ratio-1)
                if(d_var + e_var< min_v):
                    min_v = d_var + e_var
                    print min_v
                    valid_index = j
                    is_valid = True
                    #print("valid_index",valid_index)
        if (is_valid): 
            #pairs[i][valid_index] = 1#confirm the key and value the unique
            if(pairs.has_key(valid_index)):
                print("min_v",min_v)
                if(pairs[valid_index][1] > min_v):
                    pairs[valid_index] = (i,min_v)
            else:
                pairs[valid_index] = (i,min_v)
    if DEBUG: 
        print("pairs")
        print(pairs)
    
    if(len(pairs) == 0 ):
        return False, []
    else: 
        for (key,value) in pairs.items():
            key_pair.append([short[value[0]],long[key]])
        return True , key_pair

def findPossibleSen(allp, image_boxes, key_index, p1, p2):
    center = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
    #sentence = {}
    dis_p2p = {}
    avg_box = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,((image_boxes[key_index[0]][2]+image_boxes[key_index[1]][2])/2),
    ((image_boxes[key_index[0]][3]+image_boxes[key_index[1]][3])/2)]
    #sentence[key_index[0]] = p1
    #sentence[key_index[1]] = p2
    for ind, point in enumerate(allp):
        if(ind != key_index[0] and ind != key_index[1]): 
            #if(keywords_de)
            dis = tools.point_dist_to_line(p1,p2,point)
            scale_ratio = tools.compute_edge_ratio(avg_box,image_boxes[ind])
            #print("dis", dis)
            #print("dis_mesure",min(image_boxes[ind][2],image_boxes[ind][3]))
            #print("scale_ratio", scale_ratio)
            if (dis < min(image_boxes[ind][2],image_boxes[ind][3])/2 and scale_ratio<7.78 and scale_ratio >0.128):#2 rules
                #sentence[ind] = allp[ind]
                dis_p2p[ind] = tools.compute_distance(allp[ind],center)
    return dis_p2p
    
def detmKeyOrder(image_boxes,key_index,image_texts):
    ls_sentence = LinkedList.LinkedList()
    if ((image_boxes[key_index[0]][0]/(image_boxes[key_index[1]][0]+1)) < 
        (image_boxes[key_index[1]][1]/(image_boxes[key_index[0]][1]+1))):
            ls_sentence.append((image_texts[key_index[0]], key_index[0]))
            ls_sentence.append((image_texts[key_index[1]], key_index[1]))
            box1 = image_boxes[key_index[0]]  #two board char in a  sentence 
            box2 = image_boxes[key_index[1]]
            #k1 = allp[key_index[0]]
            #k2 = allp[key_index[1]]
    else:
        '''the left is zhong, the right is guo'''
        ls_sentence.append((image_texts[key_index[1]], key_index[1]))
        ls_sentence.append((image_texts[key_index[0]], key_index[0]))
        box1 = image_boxes[key_index[1]]
        box2 = image_boxes[key_index[0]]
        #k1 = allp[key_index[1]]
        #k2 = allp[key_index[0]]
    return ls_sentence,box1,box2
    
def extractKeySentence(ls_sentence, sorted_dis_p2p, k1, k2, box1,box2, allp, image_boxes,image_texts):
    key_center_dis = tools.compute_distance(k1,k2)
    '''
    the order is close to far from the keyword
    '''
    key_xmin = min(box1[0],box2[0])
    key_ymin = min(box1[1],box2[1])
    key_xmax = max(box1[0]+box1[2],box2[0]+box2[2])
    key_ymax = max(box1[1]+box1[3],box2[1]+box2[3])
    ls_init = LinkedList.LinkedList()
    
    for key, value in sorted_dis_p2p:
        print("key",key)
        print("value",value)
        #print("allp[key][0]",allp[key][0])
        '''
        if the keywords is not sequence, return []
        the point is not in the region(key_xmin, key_xmax, key_ymin, key_ymax)
        '''
        if (allp[key][0] > key_xmin  and allp[key][0] < key_xmax and
        allp[key][1] > key_ymin and allp[key][1] < key_ymax):
            return ls_init
        #dis_edge_ratio1 = compute_distance_edge_ratio(box1, image_boxes[key])
        #dis_edge_ratio2 = compute_distance_edge_ratio(box2, image_boxes[key])
        dis1 = tools.compute_distance(k1, allp[key])
        dis2 = tools.compute_distance(k2,allp[key])
        
        #print("dis_edge_ratio1",str(dis_edge_ratio1))
        #print("dis_edge_ratio2",str(dis_edge_ratio2))
        
        if (dis1 < dis2):
            #closer to zhong
            #dis_edge_ratio = dis_edge_ratio1
            #if (dis_edge_ratio > 0.169 and dis_edge_ratio < 3):
            dis_ratio = float(dis1) /float(key_center_dis)
            scale_ratio = float(image_boxes[key][2]*image_boxes[key][3]) / float(box1[2]*box2[3])
            
            if dis_ratio < 2 and dis_ratio > 0.5 and scale_ratio <2 and scale_ratio > 0.5:
                ls_sentence.insert(0,(image_texts[key],key))
                box1 = image_boxes[key]
                k1 = allp[key]
                
        else:
            #dis_edge_ratio = dis_edge_ratio2
            #if (dis_edge_ratio > 0.169 and dis_edge_ratio < 3):
            dis_ratio = float(dis2) /float(key_center_dis)
            scale_ratio = float(image_boxes[key][2]*image_boxes[key][3]) / float(box2[2]*box2[3])
            
            if dis_ratio <= 2 and dis_ratio >= 0.5 and scale_ratio <2 and scale_ratio > 0.5:
                ls_sentence.append((image_texts[key],key))
                box2 = image_boxes[key]
                k2 = allp[key]
                
    return ls_sentence
                
    #print("dis_ratio",dis_ratio)
    #print("scale_ratio",scale_ratio)
    
    
def findSentence_kernel(allp, image_boxes, image_texts, key_index):
    print('index',key_index)
    #print('allp',allp)
    p1 = allp[key_index[0]]
    p2 = allp[key_index[1]]
    

    keywords_de = tools.compute_distance_edge_ratio(image_boxes[key_index[0]],image_boxes[key_index[1]])

    ls_sentence = LinkedList.LinkedList()
    
    line1 = p1.copy()
    line2 = p2.copy()
    #sentence = {}
    dis_p2p = {}
    #sentence[key_index[0]] = p1
    #sentence[key_index[1]] = p2
    
    avg_box = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,((image_boxes[key_index[0]][2]+image_boxes[key_index[1]][2])/2),
    ((image_boxes[key_index[0]][3]+image_boxes[key_index[1]][3])/2)]
    
    
    # avg_scale = ((image_boxes[key_index[0]][2]+image_boxes[key_index[1]][2])/2) *
    #        ((image_boxes[key_index[0]][3]+image_boxes[key_index[1]][3])/2)
    count = 0
    while(count < 2):
        print("line1",line1)
        print("line2",line2)
        count += 1    
        dis_p2p = findPossibleSen(allp, image_boxes, key_index, line1, line2)
        print("dis.....................")
        ''' sort according the point to the key_center'''
        sorted_dis_p2p = sorted(dis_p2p.iteritems(), key=lambda d:d[1])
        print("the length preliminary sentence is ", str(len(sorted_dis_p2p)))
        '''
        the order is zhong guo
        '''
        ls_sentence,box1,box2 = detmKeyOrder(image_boxes,key_index,image_texts)
        
        if DEBUG:
            pre = ls_sentence.head
            while pre:
                print(pre.data)
                pre = pre.nex
        '''filter  '''
  
        #k1 = tools.getCenter(box1)
        #k2 = tools.getCenter(box2)
        k1 = [box1[0]+box1[2]/2,box1[1]+box1[3]/2]
        k2 = [box2[0]+box2[2]/2,box2[1]+box2[3]/2]
        
        ls_sentence = extractKeySentence(ls_sentence, sorted_dis_p2p, k1, k2, box1,box2, allp, image_boxes, image_texts)
        
        if (ls_sentence.is_empty()):
            return LinkedList.LinkedList()
        
        if(count < 2):
            print("********************",count)
            x_center = []
            y_center = []
            pre = ls_sentence.head
            while pre:
                index = pre.data[1]
                x_center.append(allp[index][0])
                y_center.append(allp[index][1])
                pre = pre.nex
            print("xxxxxxxxxxxxxxxxxx")
            print(x_center)
            print("yyyyyyyyyyyyyyyyyy")
            print(y_center)
            A1, B1 = optimize.curve_fit(f_1, x_center, y_center)[0]
            line1 = np.array([0, B1])
            line2 =np.array([10, 10*A1+B1])
            ls_sentence.clear()
            
    #filter guozhong
    '''    
    pre = ls_sentence.head
    only_key = True
    reverse_key = False
    while pre.nex:
        if (re.match(u'^\u56fd', pre.data)):
            #guo
            if(re.match(u'^\u4e2d', pre.nex.data)):
                #zhong
                reverse_key = True
        if ((not re.match(u'^\u56fd', pre.data)) and (not re.match(u'^\u4e2d', pre.data))):
            only_key = False
        pre = pre.nex
        '''
    #if (only_key and reverse_key):
        # print("guozhong")
        #print(len(ls_init))
    #    return ls_init
    return ls_sentence
            
def findSentence(image_boxes, image_texts, key_indexs):
    allp = tools.getCenter(image_boxes)
    sentence_all = []
    for key_index in key_indexs:
        ls_sentence = findSentence_kernel(allp,image_boxes,image_texts,key_index)
        if ( not ls_sentence.is_empty()):
            sentence_all.append(ls_sentence)
    return sentence_all
        
def readDet(source_path):    
    with open(source_path) as f:
        lines = f.read().splitlines()
        
    images_boxes = defaultdict(list)
    images_texts = defaultdict(list)
    images_scores = defaultdict(list)
    for line in lines:
        det = json.loads(line.strip())
        filename = det['image_id']+'.jpg'
        for block in det['detections']:
            score = block['score']
            
           # print("score > 0.8", score > 0.8)
            if score > 0.8:
                print("score",score)
                box = block['bbox']
                text = block['text']
                images_boxes[filename].append(box)
                images_texts[filename].append(text)
                images_scores[filename].append(score)

    return images_boxes, images_texts, images_scores

def main_from_det(source_path):

        print("loading detection")
        images_boxes, images_texts, images_scores = readDet(source_path)
        print("end loading detection")
        
        for keys in images_boxes.keys():
            #
            #if (keys != "1001283.jpg"):
            #   continue
            #print(keys)
            image_texts = images_texts[keys]
            image_boxes = images_boxes[keys]
            judge, key_indexs = findKeywords(image_boxes, image_texts)
            
            if not os.path.isdir(result):
                os.mkdir(result)
            fp2 = codecs.open(os.path.join(result,"detect_keys_sentence"),"a+","utf-8")
            fp1 = codecs.open(os.path.join(result,"detect_keys_images"),"a+","utf-8")
            
            print("after findkeyword:",judge)
            if (judge):
                for key_index in key_indexs:
                    fp1.write(keys+ " ")   
                    fp1.write(image_texts[key_index[0]]+image_texts[key_index[1]])
                    fp1.write('\n')
                    
                sentence_all = findSentence(image_boxes,image_texts,key_indexs)
                
                indexes = []
                for s in sentence_all:
                    if(len(sentence_all) != len(key_indexs)):
                        fp2.write("value"+ " ")
                    fp2.write(keys+ " ")
                    pre = s.head
                    while pre:
                        print(pre.data[0])
                        fp2.write(pre.data[0])
                        indexes.append(pre.data[1])
                        pre = pre.nex
                    fp2.write('\n')
                    '''visable'''
                    
                vis = 0
                if vis:
                    print("visable")
                    
                    if len(indexes) > 0:
                        dir_path = "/home/wudao/ctw/images/test/"
                        save_dir = "/home/wudao/ctw/images/keycentext_result_binguan/"
                        if not os.path.isdir(save_dir):
                            os.mkdir(save_dir)
                                
                        boxes_key = [image_boxes[i] for i in indexes]
                        filepath = os.path.join(dir_path, keys)
                        savepath = os.path.join(save_dir, keys)
                        vis_detection(filepath, boxes_key, savepath)
       
        fp1.close()
        fp2.close()

        
def main_from_gt():
    lines = readLines()
    for line in lines:
        image_texts = []
        image_boxes = []
        anno = json.loads(line.strip())
        #if((anno['file_name'] != "0000392.jpg") and 
        #(anno['file_name'] != "0000197.jpg")):
         #   continue
        #if(anno['file_name'] != "0000449.jpg"):
        #    continue
        #print (anno['file_name'])
        result = 'results'
        if not os.path.isdir(result):
            os.mkdir(result)
        fp2 = codecs.open(os.path.join(result,"detect_keys_sentence"),"a+","utf-8")
        fp1 = codecs.open(os.path.join(result,"detect_keys_images"),"a+","utf-8")
        for instance in anno_tools.each_char(anno):
            if not instance['is_chinese']:
                continue
            image_boxes.append(instance['adjusted_bbox'])
            image_texts.append(instance['text'])
        #print (image_texts)
        
        judge, key_indexs = findKeywords(image_boxes,image_texts)
        
        print("after findkeyword:",judge)
        if (judge):
            for key_index in key_indexs:
                fp1.write(anno['file_name']+ " ")   
                fp1.write(image_texts[key_index[0]]+image_texts[key_index[1]])
                fp1.write('\n')
                
            sentence_all = findSentence(image_boxes,image_texts,key_indexs)
            
                
            for s in sentence_all:
                if(len(sentence_all) != len(key_indexs)):
                    fp2.write("value"+ " ")
                fp2.write(anno['file_name']+ " ")
                pre = s.head
                while pre:
                    print(pre.data[0])
                    fp2.write(pre.data[0])
                    pre = pre.nex
                fp2.write('\n')
       
        fp1.close()
        fp2.close()

if __name__ == '__main__':
    main_from_gt()
    #main_from_det('/home/wudao/ctw/ctw-baseline-master/ssd/products/detections.jsonl')
    
