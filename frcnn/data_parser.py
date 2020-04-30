import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np


def get_data(input_path):
    """Parse the data from txt file

    Args:
        input_path: txt file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'gate': 500}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'gate': 0}
    """
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    with open(input_path, 'r') as f:

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'train'
                else:
                    all_imgs[filename]['imageset'] = 'val'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # note the background class in not created yet
        return all_data, classes_count, class_mapping
