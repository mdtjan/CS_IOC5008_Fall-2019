from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import json
import numpy as np
import cv2

def save_mask(dir_ann='./dataset/pascal_train.json', 
              dir_train='./dataset/train_images/', 
              dir_save='./dataset/train/masks/'):

    '''Create a mask image for augmentation'''

    with open(dir_ann, 'r') as f:
        ann = json.load(f)

    im_ids = [i['image_id'] for i in ann['annotations']]
    im_ids = list(set(im_ids))
    
    errors = []

    for idx, _id in enumerate(im_ids):

        # load training annotations
        coco = COCO(dir_ann)

        # Use the key above to retrieve information of the image
        img_info = coco.loadImgs(ids=_id)


        im_filename = img_info[0]['file_name']
        H = img_info[0]['height']
        W = img_info[0]['width']
        mask = np.zeros((H,W))
        
        # Use the imgIds to find all instance ids of the image
        annids = coco.getAnnIds(imgIds=_id)

        anns = coco.loadAnns(annids)
        
        category_ids = []
        for _ann in anns:
            temp_mask = coco.annToMask(_ann)
            temp_mask[temp_mask>0] = _ann['category_id']

            # validation step before addition
            non_zero_temp_mask = list(zip(np.where(temp_mask>0)[0], \
                np.where(temp_mask>0)[1]))

            # mask
            non_zero_mask = list(zip(np.where(mask>0)[0], \
                np.where(mask>0)[1]))

            # temp_mask
            intersection = list(set(non_zero_temp_mask) & set(non_zero_mask))

            for x,y in intersection:
                temp_mask[x,y] = 0

            mask += temp_mask

            category_ids.append(_ann['category_id'])
        
        category_ids = list(set(category_ids))
        category_ids.append(0)
        mask_unique_val = np.unique(mask)

        # validation test
        try:
            for i in mask_unique_val:
                assert i in category_ids
        except:
            errors.append((img_info, mask, category_ids))
            continue
        
        res = np.zeros((H,W,3))
        res[:,:,0] = mask
        res[:,:,1] = mask
        res[:,:,2] = mask

        # use png to avoid compression
        save_filename = img_info[0]['file_name'].replace('.jpg','.png')
        
        cv2.imwrite(dir_save+save_filename, res)
    
    return errors

def unravel_mask(mask):
    mask = np.array(mask)

    # remove 0 (background); label & mask in the same order

    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    masks = []
    H,W = mask.shape

    for i in obj_ids:
        temp_im = np.zeros((H,W))
        idx_mask = np.where(mask==i)
        for x, y in zip(idx_mask[0], idx_mask[1]):
            temp_im[x,y] = i
        masks.append(temp_im)
    return masks
