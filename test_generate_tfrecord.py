import os
import PIL
import generate_tfrecord
import numpy as np
import pandas as pd
import tensorflow as tf


class CSVToTFExampleTest(tf.test.TestCase):
    def _assertProtoEqual(self, proto_field, expectation):
        proto_list = [p for p in proto_field]
        self.assertListEqual(proto_list, expectation)

    def test_csv_to_tf_example_one_bus_per_file(self):
        """Generate tf records for one bus from one file."""
        image_file_name = 'tmp_bus_image.jpg'
        image_data = np.random.rand(256, 256, 3)
        save_path = os.path.join(self.get_temp_dir(), image_file_name)
        image = PIL.Image.fromarray(image_data, 'RGB')
        image.save(save_path)

        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        bus_data = [('tmp_bus_image.jpg', 256, 256, 'bus', 64, 64, 192, 192)]
        bus_df = pd.DataFrame(bus_data, columns=column_names)

        grouped = generate_tfrecord.split(bus_df, 'filename')
        for group in grouped:
            example = generate_tfrecord.create_tf_example(group, self.get_temp_dir())
        self._assertProtoEqual(
            example.features.feature['image/height'].int64_list.value, [256])
        self._assertProtoEqual(
            example.features.feature['image/width'].int64_list.value, [256])
        self._assertProtoEqual(
            example.features.feature['image/filename'].bytes_list.value,
            [image_file_name.encode('utf-8')])
        self._assertProtoEqual(
            example.features.feature['image/source_id'].bytes_list.value,
            [image_file_name.encode('utf-8')])
        self._assertProtoEqual(
            example.features.feature['image/format'].bytes_list.value, [b'jpg'])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/xmin'].float_list.value,
            [0.25])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/ymin'].float_list.value,
            [0.25])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/xmax'].float_list.value,
            [0.75])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/ymax'].float_list.value,
            [0.75])
        self._assertProtoEqual(
            example.features.feature['image/object/class/text'].bytes_list.value,
            [b'bus'])
        self._assertProtoEqual(
            example.features.feature['image/object/class/label'].int64_list.value,
            [1])

    def test_csv_to_tf_example_multiple_buses_per_file(self):
        """Generate tf records for multiple buses from one file."""
        image_file_name = 'tmp_bus_image.jpg'
        image_data = np.random.rand(256, 256, 3)
        save_path = os.path.join(self.get_temp_dir(), image_file_name)
        image = PIL.Image.fromarray(image_data, 'RGB')
        image.save(save_path)

        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        bus_data = [('tmp_bus_image.jpg', 256, 256, 'bus', 64, 64, 192, 192),
                        ('tmp_bus_image.jpg', 256, 256, 'bus', 96, 96, 128, 128)]
        bus_df = pd.DataFrame(bus_data, columns=column_names)

        grouped = generate_tfrecord.split(bus_df, 'filename')
        for group in grouped:
            example = generate_tfrecord.create_tf_example(group, self.get_temp_dir())
        self._assertProtoEqual(
            example.features.feature['image/height'].int64_list.value, [256])
        self._assertProtoEqual(
            example.features.feature['image/width'].int64_list.value, [256])
        self._assertProtoEqual(
            example.features.feature['image/filename'].bytes_list.value,
            [image_file_name.encode('utf-8')])
        self._assertProtoEqual(
            example.features.feature['image/source_id'].bytes_list.value,
            [image_file_name.encode('utf-8')])
        self._assertProtoEqual(
            example.features.feature['image/format'].bytes_list.value, [b'jpg'])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/xmin'].float_list.value,
            [0.25, 0.375])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/ymin'].float_list.value,
            [0.25, 0.375])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/xmax'].float_list.value,
            [0.75, 0.5])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/ymax'].float_list.value,
            [0.75, 0.5])
        self._assertProtoEqual(
            example.features.feature['image/object/class/text'].bytes_list.value,
            [b'bus', b'bus'])
        self._assertProtoEqual(
            example.features.feature['image/object/class/label'].int64_list.value,
            [1, 1])

    def test_csv_to_tf_example_one_buses_multiple_files(self):
        """Generate tf records for one bus for multiple files."""
        image_file_one = 'tmp_bus_image_1.jpg'
        image_file_two = 'tmp_bus_image_2.jpg'
        image_data = np.random.rand(256, 256, 3)
        save_path_one = os.path.join(self.get_temp_dir(), image_file_one)
        save_path_two = os.path.join(self.get_temp_dir(), image_file_two)
        image = PIL.Image.fromarray(image_data, 'RGB')
        image.save(save_path_one)
        image.save(save_path_two)

        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        bus_data = [('tmp_bus_image_1.jpg', 256, 256, 'bus', 64, 64, 192, 192),
                        ('tmp_bus_image_2.jpg', 256, 256, 'bus', 96, 96, 128, 128)]
        bus_df = pd.DataFrame(bus_data, columns=column_names)

        grouped = generate_tfrecord.split(bus_df, 'filename')
        for group in grouped:
            if group.filename == image_file_one:
                example = generate_tfrecord.create_tf_example(group, self.get_temp_dir())
                self._assertProtoEqual(
                    example.features.feature['image/height'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/width'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/filename'].bytes_list.value,
                    [image_file_one.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/source_id'].bytes_list.value,
                    [image_file_one.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/format'].bytes_list.value, [b'jpg'])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmin'].float_list.value,
                    [0.25])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymin'].float_list.value,
                    [0.25])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmax'].float_list.value,
                    [0.75])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymax'].float_list.value,
                    [0.75])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/text'].bytes_list.value,
                    [b'bus'])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/label'].int64_list.value,
                    [1])
            elif group.filename == image_file_two:
                example = generate_tfrecord.create_tf_example(group, self.get_temp_dir())
                self._assertProtoEqual(
                    example.features.feature['image/height'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/width'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/filename'].bytes_list.value,
                    [image_file_two.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/source_id'].bytes_list.value,
                    [image_file_two.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/format'].bytes_list.value, [b'jpg'])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmin'].float_list.value,
                    [0.375])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymin'].float_list.value,
                    [0.375])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmax'].float_list.value,
                    [0.5])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymax'].float_list.value,
                    [0.5])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/text'].bytes_list.value,
                    [b'bus'])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/label'].int64_list.value,
                    [1])

    def test_csv_to_tf_example_multiple_buses_multiple_files(self):
        """Generate tf records for multiple buses for multiple files."""
        image_file_one = 'tmp_bus_image_1.jpg'
        image_file_two = 'tmp_bus_image_2.jpg'
        image_data = np.random.rand(256, 256, 3)
        save_path_one = os.path.join(self.get_temp_dir(), image_file_one)
        save_path_two = os.path.join(self.get_temp_dir(), image_file_two)
        image = PIL.Image.fromarray(image_data, 'RGB')
        image.save(save_path_one)
        image.save(save_path_two)

        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        bus_data = [('tmp_bus_image_1.jpg', 256, 256, 'bus', 64, 64, 192, 192),
                        ('tmp_bus_image_1.jpg', 256, 256, 'bus', 32, 32, 96, 96),
                        ('tmp_bus_image_2.jpg', 256, 256, 'bus', 96, 96, 128, 128)]
        bus_df = pd.DataFrame(bus_data, columns=column_names)

        grouped = generate_tfrecord.split(bus_df, 'filename')
        for group in grouped:
            if group.filename == image_file_one:
                example = generate_tfrecord.create_tf_example(group, self.get_temp_dir())
                self._assertProtoEqual(
                    example.features.feature['image/height'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/width'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/filename'].bytes_list.value,
                    [image_file_one.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/source_id'].bytes_list.value,
                    [image_file_one.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/format'].bytes_list.value, [b'jpg'])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmin'].float_list.value,
                    [0.25, 0.125])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymin'].float_list.value,
                    [0.25, 0.125])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmax'].float_list.value,
                    [0.75, 0.375])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymax'].float_list.value,
                    [0.75, 0.375])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/text'].bytes_list.value,
                    [b'bus', b'bus'])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/label'].int64_list.value,
                    [1, 1])
            elif group.filename == image_file_two:
                example = generate_tfrecord.create_tf_example(group, self.get_temp_dir())
                self._assertProtoEqual(
                    example.features.feature['image/height'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/width'].int64_list.value, [256])
                self._assertProtoEqual(
                    example.features.feature['image/filename'].bytes_list.value,
                    [image_file_two.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/source_id'].bytes_list.value,
                    [image_file_two.encode('utf-8')])
                self._assertProtoEqual(
                    example.features.feature['image/format'].bytes_list.value, [b'jpg'])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmin'].float_list.value,
                    [0.375])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymin'].float_list.value,
                    [0.375])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/xmax'].float_list.value,
                    [0.5])
                self._assertProtoEqual(
                    example.features.feature['image/object/bbox/ymax'].float_list.value,
                    [0.5])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/text'].bytes_list.value,
                    [b'bus'])
                self._assertProtoEqual(
                    example.features.feature['image/object/class/label'].int64_list.value,
                    [1])
