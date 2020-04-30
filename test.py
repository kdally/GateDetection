from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import os, csv, cv2, pickle, time, wget
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.utils import Progbar


from frcnn import roi_helpers
import frcnn.resnet as nn


def load_weights(im_size, model_rpn, model_classifier):
    """ load or downloads the pre-trained weight on a generic dataset or on the WashingtonOB Race dataset"""

    if not os.path.isfile(f'models/model_frcnn_{im_size}.hdf5'):
        print('Model weights non-existent yet. Downloading pre-trained weights. Please wait.')
        url = f'https://srv-file14.gofile.io/download/Wbj1ty/model_frcnn_{im_size}.hdf5'

        try:
            wget.download(url=url, out=f'models/model_frcnn_{im_size}.hdf5')
            print('Done downloading.')
            model_rpn.load_weights(f'models/model_frcnn_{im_size}.hdf5', by_name=True)
            model_classifier.load_weights(f'models/model_frcnn_{im_size}.hdf5', by_name=True)

        except:
            print('Download failed. Please download manually from '
                  f'{url} and place the file in the "models" directory.')
            exit()

    else:
        model_rpn.load_weights(f'models/model_frcnn_{im_size}.hdf5', by_name=True)
        model_classifier.load_weights(f'models/model_frcnn_{im_size}.hdf5', by_name=True)
        print(f'Loading weights from models/model_frcnn_{im_size}.hdf5')

    return model_rpn, model_classifier


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def get_real_coordinates(ratio, x1, y1, x2, y2):
    """ transforms the coordinates of the bounding box to its original size """
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def test(im_size=600, mode='normal', detect_threshold=0.9, overlap_threshold=0.5):
    """ Test the model
    Args:
        im_size: trained model available for 300, 400 or 600. Input images are resized to this size.
        mode: 'normal' means predictions will be saved. Any other mode means only a CSV corner file will be saved.
        detect_threshold: minimum class belonging probability for a proposal to be accepted
        overlap_threshold: maximum IoU between two proposals

    Returns:
        avg_time: time taken over the last 10 images. Time is measured from loading the image to saving the predictions.
    """

    conf_file = f'models/conf/config_frcnn_{im_size}.pickle'
    C = pickle.load(open(conf_file, 'rb'))

    img_path = 'data/testing/images'

    class_mapping = C.class_mapping

    class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.array([0, 128, 255]) for v in class_mapping}

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base resnet network
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn, model_classifier = load_weights(im_size, model_rpn, model_classifier)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    output_folder = f'output/predictions/frcnn_size{int(im_size)}_p{int(detect_threshold * 100)}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = open(f'output/predictions/frcnn_size{int(im_size)}_p{int(detect_threshold * 100)}/'
                       f'predicted_corners_size{int(im_size)}_p{int(detect_threshold * 100)}.csv', 'w')
    writer = csv.writer(output_file)

    print(
        f'Predicting gates for im_size={im_size} and detection probability threshold of {int(detect_threshold * 100)}%.')
    print(
        f'Output to be saved in directory "/output/predictions/frcnn_size{int(im_size)}_p{int(detect_threshold * 100)}/"')
    progbar = Progbar(len(os.listdir(img_path)) - 1)

    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if not img_name.lower().endswith(('.png', '.jpg')):
            continue

        filepath = os.path.join(img_path, img_name)

        start_time = time.time()
        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, overlap_thresh=overlap_threshold, max_boxes=C.max_boxes)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        time_list = []

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < detect_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # only gate objects
                cls_name = 'gate'

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                # only gate objects, which is index 0
                cls_num = 0

                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass

                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                        overlap_thresh=overlap_threshold)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                w = (x2 - x1)
                h = (y2 - y1)
                scale = 0.16
                x1 += w * scale
                x2 -= w * scale
                y1 += h * scale
                y2 -= h * scale

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                writer.writerow([img_name, real_x1, real_y1, real_x2, real_y1, real_x2, real_y2, real_x1, real_y2])

                if mode == 'normal':
                    cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                                  (int(class_to_color[key][0]), int(class_to_color[key][1]),
                                   int(class_to_color[key][2])),
                                  2)

                    textLabel = f'{key}: {int(100 * new_probs[jk])}%'

                    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    textOrg = (real_x1, real_y1)

                    cv2.rectangle(img, (textOrg[0], textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), 2)
                    cv2.rectangle(img, (textOrg[0], textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 128, 255), -1)
                    cv2.putText(img, textLabel, (real_x1, real_y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)

        time_taken = time.time() - start_time
        progbar.update(idx)
        print(f'   -   Elapsed time: {time_taken:.3}s for {img_name}')
        if idx > 20:
            time_list.append(time_taken)

        if mode == 'normal':
            cv2.imwrite(
                f'output/predictions/frcnn_size{int(im_size)}_p{int(detect_threshold * 100)}/predict_{img_name}', img)
            plt.close()

    output_file.close()

    # return the prediction time over the last 10 images
    return sum(time_list) / len(time_list)

# test(im_size=600, detect_threshold=0.6, overlap_threshold=0.6)
