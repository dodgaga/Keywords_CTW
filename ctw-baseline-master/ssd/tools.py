import numpy as np


def point_dist_to_line(p1,p2,p3):
    #computer the distance from p3 to p1 - p2
    return np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2 - p1)
    
def compute_distance_edge_ratio(box1,box2):
    cor1 = []
    cor1.append(box1[0]+box1[2]/2)
    cor1.append(box1[1]+box1[3]/2)
    
    cor2 = []
    cor2.append(box2[0]+box2[2]/2)
    cor2.append(box2[1]+box2[3]/2)
    dis = np.sqrt(np.sum((np.array(cor1) - np.array(cor2))**2))
    return dis/max((box1[2]+box2[2])/2, (box1[3]+box2[3])/2)

def compute_distance(p1,p2):
    dis = np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
    return dis

def compute_edge_ratio(box1,box2):
    width_compare = box1[2]/box2[2]
    height_compare = box1[3]/box2[3]
    return width_compare*height_compare
    
def getCenter(image_boxes):
    image_boxes = np.array(image_boxes)
    print("image_boxes",image_boxes)
    #x_center = image_boxes[:][0]+image_boxes[:][2]/2
    x_center = [(x[0]+x[2]/2) for x in image_boxes]
    y_center = [(x[1]+x[3]/2) for x in image_boxes]
    return np.stack((x_center,y_center),axis=-1)