import imgaug as ia
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
from collections import namedtuple
import glob

PATH_TO_ANNO = '../annotations/'
ANNO_TABLE = 'bus_labels.csv'
PATH_TO_IMAGES = '../images/'
PATH_TO_RESIZED_IMAGES = '../images/resized/'
PATH_TO_AUG_IMAGES = 'images/'


def get_images_bboxes(df=None, path_to_images=None, head=None):
    """
    the function get a list of images and bboxes
    and print them side by side
    """
    if not path_to_images:
        path_to_images = PATH_TO_AUG_IMAGES

    # make a list of subgroups, each subgroup is of one file
    data = namedtuple('data', ['file', 'object'])
    grouped = df.groupby('filename')
    gd_data = []
    file_naems = []
    for file, x in zip(grouped.groups.keys(), grouped.groups):
        gd_data.append(data(file, grouped.get_group(x)))
        file_naems.append(file)

    filenames = []
    images = []
    bboxes = []
    for g in gd_data:
        grouped_image = g.object
        H = int(grouped_image['height'].iloc[0])
        W = int(grouped_image['width'].iloc[0])
        filename = grouped_image['filename'].iloc[0]

        # create list of bboxes object to each image
        bboxes_on_an_image = []
        for idx, row in grouped_image.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            bboxes_on_an_image.append(ia.BoundingBox(x1, y1, x2, y2))

        bboxes.append(ia.BoundingBoxesOnImage(
                                bboxes_on_an_image, shape=(H, W)))
        filenames.append(filename)

        # open an image
        path = os.path.join(path_to_images, filename)
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        images.append(image)

    return filenames, images, bboxes


def imshow_bbox(images, bboxes, head=None):
    """
    plot images with bboxes
    """
    GREEN = [0, 255, 0]
    color = GREEN

    for i, (img, bb) in enumerate(zip(images, bboxes)):
        bboxes_cut = bb.remove_out_of_image().cut_out_of_image()
        image_bbox = bboxes_cut.draw_on_image(img, thickness=2, color=color)
        plt.figure()
        plt.imshow(image_bbox)
        if head and (i+1 == head):
            break


def imresize_to_300x225(path_to_images, output_path):
    H = 225
    W = 300
    images_add = glob.glob(path_to_images + '*.JPG')
    for add in images_add:
        temp = cv.imread(add)
        temp = cv.resize(temp, (W, H))
        new_add = os.path.join(
            output_path, 's_' + add[add.find('\\')+1:])
        cv.imwrite(new_add, temp)


def scale_anno(df):
    H = 225
    W = 300
    Hi = 2736
    Wi = 3648
    scale_x = lambda x: int(x*(W/Wi))
    scale_y = lambda y: int(y*(H/Hi))
    df[['xmin', 'xmax']] = df[['xmin', 'xmax']].applymap(scale_x)
    df[['ymin', 'ymax']] = df[['ymin', 'ymax']].applymap(scale_y)
    df['height'] = H
    df['width'] = W
    df['filename'] = 's_' + df['filename']

    # filename col as a grouped list
    # return sorted(list(set(df['filename'].tolist())))


def create_anno(filenames, bboxes, W, H, aug=False):
    anno = {}
    anno['filename'] = []
    anno['xmin'] = []
    anno['ymin'] = []
    anno['xmax'] = []
    anno['ymax'] = []

    if aug:
        y_offset = 37
        f_aug = 'aug_'
    else:
        y_offset = 0
        f_aug = ''

    for i, (f, bb) in enumerate(zip(filenames, bboxes)):
        for _, box in enumerate(bb.bounding_boxes):
            anno['filename'].append(f_aug + f)
            anno['xmin'].append(box.x1)
            anno['ymin'].append(box.y1 + y_offset)
            anno['xmax'].append(box.x2)
            anno['ymax'].append(box.y2 + y_offset)

    df = pd.DataFrame(anno)
    df.insert(1, 'width', W)
    df.insert(2, 'height', H)
    df.insert(3, 'class', 'bus')

    return df


def imwrite_aug_ssd(images_aug, filenames, output_path):
    """
    write augmeted images to output_path
    add zero padding to fit H=300
    """
    for i, (filename, image_after) in enumerate(zip(filenames, images_aug)):
        temp = cv.copyMakeBorder(image_after,37,38,0,0,cv.BORDER_CONSTANT)
        temp = cv.cvtColor(temp, cv.COLOR_BGR2RGB)
        file = os.path.join(PATH_TO_AUG_IMAGES, 'aug_' + filename)
        cv.imwrite(file, temp)
