import numpy as np
import os
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2 as cv
import time

PATH_TO_IMAGES = './images/ssd_resized/'
MODEL_PATH = './models/ssd/all/rev2/'

def run_inference(images_path=PATH_TO_IMAGES, model_path=MODEL_PATH):
    
    NUM_CLASSES = 6

    # images 
    test_images = glob.glob(images_path + '*JPG')
    N = len(test_images)
    W, H = 3648, 2736

    # load and resize test images
    input_images = []
    Wr, Hr = 300, 225   # resize shape
    for path in test_images:
        image = cv.imread(path)
        image = cv.resize(image, (Wr, Hr), interpolation=cv.INTER_AREA)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        input_images.append(image)

    # load frozen model
    PATH_TO_CKPT = os.path.join(model_path, 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    # start inference session
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            tic = time.time()
            outputs = sess.run(tensor_dict, feed_dict={image_tensor: input_images})
            toc = time.time()


    # create annotation file in the define format
    anno_filename = 'annotationsTest.txt'
    anno_file = open(anno_filename, 'w')

    # parse each detection to a line and write to file
    for i, (det_score, det_cls, det_box, img_path) in enumerate(zip(outputs['detection_scores'], outputs['detection_classes'],
                                                                    outputs['detection_boxes'], test_images)):
        # define object score threshold
        obj_indices = np.where(det_score > 0.3)
        # if there is no score that passes the threshold pick the max
        if not obj_indices[0].size:
            obj_indices = [np.argmax(det_score)]
        # pickup winner classes
        winner_classes = det_cls[obj_indices].astype(int)
        # pickup winner classes' bboxes
        winner_boxes = np.rint(np.multiply(det_box, [H, W, H, W])[obj_indices]).astype(int)
        filename = img_path[img_path.rfind('_')+1:]
        line = '{}:'.format(filename)
        for c, b in zip(winner_classes, winner_boxes):
            line = line + '[{},{},{},{},{}],'.format(b[1], b[0], b[3]-b[1], b[2]-b[0], c)
        line = line[:-1] + '\n'
        anno_file.write(line)

    anno_file.close()

    print('\n\n\n\nIt took only {:.3f}[sec] to infer {} images!'.format(toc-tic, N))
    print('Annotation saved at {}/{}\n\n\n\n'.format(os.getcwd(), anno_filename))


if __name__ == "__main__":
    run_inference()
