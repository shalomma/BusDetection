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


def imresize_to_300x225(path_to_images, output_path, df):
    scale_anno(df)
    H = 225
    W = 300
    images_add = glob.glob(path_to_images + '*.JPG')
    if images_add:
        return

    for add in images_add:
        temp = cv.imread(add)
        temp = cv.resize(temp, (W, H))
        new_add = os.path.join(
            output_path, 's_' + add[add.find('\\')+1:])
        cv.imwrite(new_add, temp)


def scale_anno(df):
    """
    scale the DF to fit ssd
    """
    H = 225
    W = 300
    Hi = 2736
    Wi = 3648

    def scale_x(x):
        return int(x*(W/Wi))
    df[['xmin', 'xmax']] = df[['xmin', 'xmax']].applymap(scale_x)

    def scale_y(y):
        return int(y*(H/Hi))
    df[['ymin', 'ymax']] = df[['ymin', 'ymax']].applymap(scale_y)

    df['height'] = H
    df['width'] = W
    df['filename'] = 's_' + df['filename']

    # filename col as a grouped list
    # return sorted(list(set(df['filename'].tolist())))


def create_anno(images, bboxes, path_to_images, filenames=None, ssd=False):
    """
    return a DF: col[filename,w,h,class,x1,y1,x2,y2]
    if ssd: make adjustments to fit the written images
    """
    anno = {}
    anno['filename'] = []
    anno['width'] = []
    anno['height'] = []
    anno['class'] = []
    anno['xmin'] = []
    anno['ymin'] = []
    anno['xmax'] = []
    anno['ymax'] = []
    anno['path'] = []

    if filenames:
        filenames_o = filenames
    else:
        images_add = glob.glob(path_to_images + '*.JPG')
        filenames_o = [s[s.find('\\')+1:] for s in images_add]

    for i, (f, bb, img) in enumerate(zip(filenames_o, bboxes, images)):
        for _, box in enumerate(bb.bounding_boxes):
            anno['filename'].append(f)
            anno['xmin'].append(box.x1)
            anno['ymin'].append(box.y1)
            anno['xmax'].append(box.x2)
            anno['ymax'].append(box.y2)
            anno['width'].append(img.shape[1])
            anno['height'].append(img.shape[0])
            anno['path'].append(path_to_images)
            anno['class'].append('bus')

    df = pd.DataFrame(anno)
    if ssd:
        df['width'] = 300
        df['height'] = 300

        def scale_y(y):
            return int(y + 37)
        df[['ymin', 'ymax']] = df[['ymin', 'ymax']].applymap(scale_y)

        def prefix_aug(f):
            return str('aug_' + f)
        df[['filename']] = df[['filename']].applymap(prefix_aug)

    df = df[['filename', 'width', 'height', 'class',
            'xmin', 'ymin', 'xmax', 'ymax', 'path']]

    return df


def imwrite_images_to_path(images, filenames, output_path, ssd=False):
    """
    write augmeted images to output_path
    add zero padding to fit H=300
    """
    if ssd:
        top_offset = 37
        bottom_offset = 38
        prefix_aug = 'aug_'
    else:
        top_offset = 0
        bottom_offset = 0
        prefix_aug = ''

    filenames = [prefix_aug + f for f in filenames]
    for i, (filename, img) in enumerate(zip(filenames, images)):
        temp = cv.copyMakeBorder(img, top_offset, bottom_offset,
                                 0, 0, cv.BORDER_CONSTANT)
        temp = cv.cvtColor(temp, cv.COLOR_BGR2RGB)
        file = os.path.join(PATH_TO_AUG_IMAGES, filename)
        cv.imwrite(file, temp)


def imwrite_aug_ssd(images, filenames, bboxes,
                    output_images, output_csv=None):
    """
    1. padd augmented images and save them to disk
    2. create an annoteation DF
    3. save DF to a csv file
    """

    imwrite_images_to_path(images, filenames,
                           output_path=output_images, ssd=True)
    df = create_anno(images, bboxes,
                     path_to_images=output_images,
                     filenames=filenames, ssd=True)
    return df
