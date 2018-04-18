# -*- coding:utf-8 -*-
import Levenshtein
import chardet
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import defaultdict  
def bjjl(a,b):
    try:
        if chardet.detect(a)['encoding'] == 'utf-8':
            a = a.decode('utf-8')
        if chardet.detect(b)['encoding'] == 'utf-8':
            b = b.decode('utf-8')
    finally:
        return Levenshtein.distance(a,b)

def getSentence(filename):
    file = open(filename)
    dict = defaultdict(list) 
    for line in file:
        pair = line.split()
        temp = dict[pair[0]]
        temp.append(pair[1])
        dict[pair[0]] = temp        
    file.close()
    return dict  
    
def compare(gt_path, detect_path):
    dict_gt = getSentence(gt_path)
    #print("gt_len",len(dict_gt))
    dict_det = getSentence(detect_path)
    
    normolize_dis_all = 0
    count = 0
    len_gt = 0
    for key in dict_gt:
        print(key)
        len_gt += len(dict_gt[key])
        if dict_det.has_key(key):
            min_dis = sys.maxint
            for s in dict_gt[key]:
                for ans in dict_det[key]:
                    dis = bjjl(s, ans)
                    if dis < min_dis:
                        min_dis = dis
                if min_dis == 0:
                    count += 1
                normolize_dis = float(min_dis)/float(len(s))
                print(normolize_dis)
                normolize_dis_all += normolize_dis
        else:
            normolize_dis_all += len(dict_gt[key])
    
            
    return normolize_dis_all,count,len_gt
    
if __name__ == '__main__':
    '''
    a = '你好abc'
    b = '您好ab'
    print(bjjl(a,b))
    c = u'abc你好'
    d = u'ab您好'
    print(bjjl(c,d))
    '''
    gt = sys.argv[1]
    detect = sys.argv[2]
    dis, true_count, len_gt = compare(gt,detect)
    print("all_distance", dis )
    print("true_count", true_count )
    print("len_gt", len_gt )
    print("accuracy",float(true_count)/float(len_gt))
    print("ED",dis/len_gt)
