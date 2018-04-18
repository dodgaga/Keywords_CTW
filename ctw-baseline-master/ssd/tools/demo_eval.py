# -*- coding:utf-8 -*-
import sys
def getAnswer(filename):
    file = open(filename)
    dict = {}
    i = 0
    for line in file:
        pair = line.split()
        if dict.has_key(pair[0]):
            dict[pair[0]] += 1
        else:
            dict[pair[0]] = 1
    file.close()
    return dict
if __name__ == '__main__':
    gt = getAnswer(sys.argv[1])
    ans = getAnswer(sys.argv[2])
    tp = 0
    all = 0
    imageTp = 0
    imageNum = 0
    imageDet = 0
    imageError = 0
    result = open('result.txt', 'w')
    for key in gt:
        if ans.has_key(key):
            imageDet += 1
            tp += min(gt[key], ans[key])
            if gt[key] == ans[key]:
                imageTp += 1
            elif (gt[key] > ans[key]):
                result.write("less "+key + "\n")
            else:
                result.write("more "+key + "\n")
        else:
            result.write("loujian "+key + "\n")
            #imageError += 1
        imageNum += 1 
        all += gt[key]
    result.close()
    
    recall = float(imageDet)/float(imageNum)
    precision = float(imageDet)/(float(len(ans)))
    
    f_measure = 2*recall*precision / (recall+precision)
    print("tp:", tp)
    print("all:", all)
    print("imageTp:", imageTp)
    print("imageNum:", imageNum)
    print("imageDet",imageDet)
    print("imageError",len(ans)-imageDet)
    print ("recall", str(float(imageDet)/float(imageNum)))
    print ("precision", str(float(imageDet)/(float(len(ans)))))
    print('f_measure',str(f_measure))
   # print("accuracy",str(float(tp)/float(all)))
