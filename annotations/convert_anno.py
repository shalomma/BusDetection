import pandas as pd
import glob
import os


def convert_anno(original_anno_path='./', output_path='./'):
    """
    Convert the given annotation file format
    to fit the generate_tfrecord script
    :param original_anno_path: a path to the given annotation file (txt)
    :param output_path: a path to where the new csv file will be generated
    """
    file = (glob.glob(os.path.join(original_anno_path, '*txt')))[0]
    ftxt = open(file, "r")
    fcsv = open("temp.csv", "w+")

    fcsv.writelines("filename,xmin,ymin,w,h,class\n")

    for line in ftxt.readlines():
        cl = '['
        cr = ']'
        obj = line.count(cl)
        c_list = [pos for pos, char in enumerate(line)
                  if (char == cl or char == cr)]
        for i in range(obj):
            fcsv.writelines(line[0:12] + "," +
                            line[c_list[2*i]+1:c_list[2*i+1]] + "\n")

    ftxt.close()
    fcsv.close()

    df = pd.read_csv('temp.csv')
    df[['xmin', 'ymin', 'w', 'h']] = \
        df[['xmin', 'ymin', 'w', 'h']].astype(dtype=int)
    df['xmax'] = df['xmin'] + df['w']
    df['ymax'] = df['ymin'] + df['h']
    df['width'] = 3648
    df['height'] = 2736
    df = df[['filename', 'width', 'height', 'class',
             'xmin', 'ymin', 'xmax', 'ymax']]
    new_file = os.path.join(output_path, 'annotationsTrain.csv')
    print("Done.\nAnnotation CSV file is ready at: {}annotationsTrain.csv".
          format(output_path))
    os.remove("temp.csv")
    df.to_csv(new_file, index=False)


if __name__ == "__main__":
    # execute only if run as a script
    convert_anno()
