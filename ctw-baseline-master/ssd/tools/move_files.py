import shutil,os
with open('/home/wudao/ctw/ctw-baseline-master/ssd/temp.txt') as f:
    lines = f.read().splitlines()

for line in lines:

    images_name = line.split(' ')[0]

    image_path = os.path.join('/home/wudao/ctw/images/trainval/',images_name)
    target_path = os.path.join('/home/wudao/select_img/',images_name)
    shutil.copyfile(image_path,target_path)
    