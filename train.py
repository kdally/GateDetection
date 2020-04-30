from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import os, random, time, pickle, wget, math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

from frcnn import config, regions_proposal_network
from frcnn import losses as losses
import frcnn.roi_helpers as roi_helpers
from frcnn.data_parser import get_data
from frcnn import resnet as nn


def load_weights(im_size, model_rpn, model_classifier, train_mode):
    """ load or downloads the pre-trained weight on a generic dataset or on the WashingtonOB Race dataset.
        Create log file for this model.
    """

    if train_mode == 1:
        url = f'https://srv-file12.gofile.io/download/CW10qJ/model_frcnn.hdf5'
        model_path = f'models/model_frcnn.hdf5'

    if train_mode == 2:
        url = f'https://srv-file14.gofile.io/download/Wbj1ty/model_frcnn_{im_size}.hdf5'
        model_path = f'models/model_frcnn_{im_size}.hdf5'

    if not os.path.isfile(model_path):
        print('Model weights non-existent yet. Downloading pre-trained weights. Please wait.')

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
        model_rpn.load_weights(model_path, by_name=True)
        model_classifier.load_weights(model_path, by_name=True)
        print(f'Loading weights from {model_path}.hdf5')

    # Create the record.csv file to record losses and accuracy
    if not os.path.isfile(f'models/logs/frcnn_{im_size}_adap.csv') or train_mode == 1:
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes',
                                          'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls',
                                          'loss_class_regr',
                                          'curr_loss', 'class_acc_val', 'curr_loss_val', 'elapsed_time'])
    else:
        record_df = pd.read_csv(f'models/logs/frcnn_{im_size}_adap.csv')

    return model_rpn, model_classifier, record_df


def calculate_learning_rate(epoch, mode='adaptive'):
    """ Obtain the learning rate based on the epoch number with exponential decay"""

    if mode == 'constant':
        return 1e-5
    initial_lrate = 1e-4
    k = 1.5
    return initial_lrate * math.exp(-k * epoch)


def validate(val_steps, data_gen_val, model_rpn, C, class_mapping, model_classifier):
    """ Perform validation by evaluating both netowrks on unseen data and calculate total loss
        (regression and classification for both networks)
    """

    val_loss = np.zeros((val_steps, 5))

    progbar2 = generic_utils.Progbar(val_steps)

    for idx in range(val_steps):

        # use next to extract data since it is a generator
        X, Y, img_data = next(data_gen_val)

        # evaluate model on validation data
        loss_rpn = model_rpn.evaluate(X, Y, verbose=0)

        # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
        P_rpn = model_rpn.predict_on_batch(X)

        # R: bboxes (shape=(im_size,4))
        # Convert rpn layer to roi bboxes
        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, overlap_thresh=0.6, max_boxes=C.max_boxes)

        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        # X2: bboxes with iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
        # Y2: corresponding labels and corresponding ground truth bboxes
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

        if X2 is None:
            continue

        neg_samples = np.where(Y1[0, :, -1] == 1)  # wo letzte Klasse (bg) = 1
        pos_samples = np.where(Y1[0, :, -1] == 0)  # wo andere Klasse = 1, i.e. background = 0, da one hot

        if len(neg_samples) > 0:
            neg_samples = neg_samples[0]
        else:
            neg_samples = []

        if len(pos_samples) > 0:
            pos_samples = pos_samples[0]
        else:
            pos_samples = []

        if C.num_rois > 1:

            # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
            if len(pos_samples) < C.num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()

            # Randomly choose (num_rois - num_pos) neg samples
            try:
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                        replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                        replace=True).tolist()

            # Save all the pos and neg samples in sel_samples
            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            # in the extreme case where num_rois = 1, we pick a random pos or neg sample
            if np.random.randint(0, 2):
                sel_samples = random.choice(neg_samples)
            else:
                sel_samples = random.choice(pos_samples)

        # validation_data: [X, X2[:, sel_samples, :]]
        # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
        #  X                     => img_data resized image
        #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
        #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
        #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
        loss_class = model_classifier.evaluate([X, X2[:, sel_samples, :]],
                                               [Y1[:, sel_samples, :], Y2[:, sel_samples, :]], verbose=0)

        val_loss[idx, 0] = loss_rpn[1]
        val_loss[idx, 1] = loss_rpn[2]
        val_loss[idx, 2] = loss_class[1]
        val_loss[idx, 3] = loss_class[2]
        val_loss[idx, 4] = loss_class[3]

        progbar2.update(idx + 1)

    val = {'loss_rpn_cls': np.mean(val_loss[:, 0]), 'loss_rpn_regr': np.mean(val_loss[:, 1]),
           'loss_class_cls': np.mean(val_loss[:, 2]), 'loss_class_regr': np.mean(val_loss[:, 3]),
           'class_acc': np.mean(val_loss[:, 4])}

    val['curr_loss'] = val['loss_rpn_cls'] + val['loss_rpn_regr'] + val['loss_class_cls'] + val['loss_class_regr']

    return val


def train(im_size, train_mode, epoch_length=1000, num_epochs=10, val_steps=200, lr_mode='adaptive'):
    """ Train the model
    Args:
        im_size: user input. Input images are resized to this size.
        train_mode: 1 to train new model with pre-trained weights on generic dataset.
                    2 to keep training on the WashingtonOB Race dataset.
        epoch_length: number of steps per training epoch
        num_epochs: maximum number of training epochs
        val_steps: number of validation steps, at the end of each training epoch
        lr_mode: if 'adaptive', learning rate varies with eponential decay. If 'constant', it is 1e-5.

    Returns:
       None - (trained model saved automatically)
    """

    C = config.Config()
    C.network = 'resnet50'
    C.im_size = im_size
    C.model_path = f'models/model_frcnn_{C.im_size}.hdf5'

    all_imgs, classes_count, class_mapping = get_data('data/training/corners_training.txt')

    # add background class
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    print(f'Total number of objects per class (across all dataset): {class_mapping}')

    config_output_filename = f'models/conf/config_frcnn_{C.im_size}.pickle'

    record_path = f'models/logs/frcnn_{C.im_size}.csv'  # Where to record data (used to save the losses, classification accuracy and mean average precision)
    if lr_mode == 'adaptive':
        record_path = f'models/logs/frcnn_{C.im_size}_adap.csv'

    pickle.dump(C, open(config_output_filename, 'wb'))

    train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

    # Shuffle the images with seed 1
    random.seed(1)
    random.shuffle(train_imgs)
    random.shuffle(val_imgs)

    print(f'{len(train_imgs)} and {len(val_imgs)} training and validation samples (before augmentation), respectively.')

    data_gen_train = regions_proposal_network.get_anchor_gt(train_imgs, C, nn.get_img_output_length, mode='train')
    data_gen_val = regions_proposal_network.get_anchor_gt(val_imgs, C, nn.get_img_output_length, mode='val')

    X, Y, image_data = next(data_gen_train)

    print('Original image: height=%d width=%d' % (image_data['height'], image_data['width']))
    print('Resized image:  height=%d width=%d' % (X.shape[1], X.shape[2]))
    print('Feature map size: height=%d width=%d C.rpn_stride=%d' % (Y[0].shape[1], Y[0].shape[2], C.rpn_stride))

    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (resnet here)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    model_rpn, model_classifier, record_df = load_weights(im_size, model_rpn, model_classifier, train_mode)

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # Training setting
    total_epochs = len(record_df)
    if len(record_df) == 0:
        best_loss = np.Inf
    else:
        best_loss = np.min(record_df['curr_loss_val'])
        print(f'Resuming training. Already trained for {len(record_df)} epochs.')

    validation_trend_hold = False

    total_epochs += num_epochs
    iter_num = 0

    loss = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    print('Starting training')
    start_time = time.time()

    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                # use next to extract data since it is a generator
                X, Y, img_data = next(data_gen_train)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                current_learning_rate = calculate_learning_rate(epoch_num, mode=lr_mode)
                K.set_value(model_rpn.optimizer.lr, current_learning_rate)
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)

                # R: bboxes (shape=(im_size,4))
                # Convert rpn layer to roi bboxes
                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, overlap_thresh=0.6, max_boxes=C.max_boxes)

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2: bboxes with iou > C.classifier_min_overlap for all gt bboxes in 20 non_max_suppression bboxes
                # Y2: corresponding labels and corresponding ground truth bboxes
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:

                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < C.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()

                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                K.set_value(model_classifier.optimizer.lr, current_learning_rate)
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                loss[iter_num, 0] = loss_rpn[1]
                loss[iter_num, 1] = loss_rpn[2]

                loss[iter_num, 2] = loss_class[1]
                loss[iter_num, 3] = loss_class[2]
                loss[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1,
                               [('RPN Classifier Loss', loss[iter_num, 0]), ('RPN Regression Loss', loss[iter_num, 1]),
                                ('Detector Classifier Loss', loss[iter_num, 2]),
                                ('Detector Regression Loss', loss[iter_num, 3])])

                iter_num += 1

                # end of epoch check
                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(loss[:, 0])
                    loss_rpn_regr = np.mean(loss[:, 1])
                    loss_class_cls = np.mean(loss[:, 2])
                    loss_class_regr = np.mean(loss[:, 3])
                    class_acc = np.mean(loss[:, 4])

                    print("Performing validation.")
                    val_loss = validate(val_steps, data_gen_val, model_rpn, C, class_mapping,
                                        model_classifier)

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print(f'Classifier accuracy for bounding boxes: {class_acc}')

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0

                    if val_loss['curr_loss'] <= best_loss:
                        if C.verbose:
                            print(
                                f'Total validation loss decreased from {best_loss} to {val_loss["curr_loss"]}, saving weights.')
                            print('')
                        best_loss = val_loss['curr_loss']
                        model_all.save_weights(C.model_path)
                        validation_trend_hold = False

                    elif not validation_trend_hold:
                        if C.verbose:
                            print(
                                f'Total validation loss increased for the first time, from {best_loss} to {val_loss["curr_loss"]}. Performing one more epoch to verify trend. Not saving weights for now.')
                            print('')
                            validation_trend_hold = True

                    else:
                        if C.verbose:
                            print(
                                f'Total validation loss increased for the second time, from {best_loss} to {val_loss["curr_loss"]}.')
                            print(
                                f'Terminating training now to prevent over-fitting. Keeping weights from epoch {epoch_num - 1}.')
                            exit()

                    new_row = {'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                               'class_acc': round(class_acc, 3),
                               'loss_rpn_cls': round(loss_rpn_cls, 3),
                               'loss_rpn_regr': round(loss_rpn_regr, 3),
                               'loss_class_cls': round(loss_class_cls, 3),
                               'loss_class_regr': round(loss_class_regr, 3),
                               'curr_loss': round(curr_loss, 3),
                               'class_acc_val': round(val_loss['class_acc'], 3),
                               'curr_loss_val': round(val_loss['curr_loss'], 3),
                               'elapsed_time': round(time.time() - start_time, 3)}

                    start_time = time.time()
                    record_df = record_df.append(new_row, ignore_index=True)
                    record_df.to_csv(record_path, index=False)

                    break

            except Exception as e:
                print(f'Exception: {e}')
                continue

    print('Training complete.')
    return


def plot_training_comp(type1, type2):
    record_df1 = pd.read_csv(f'models/logs/frcnn_{type1}.csv')
    record_df2 = pd.read_csv(f'models/logs/frcnn_{type2}.csv')
    record_df1['curr_loss'] = record_df1['loss_rpn_cls'] + record_df1['loss_rpn_regr'] + record_df1['loss_class_cls'] + \
                              record_df1['loss_class_regr']
    record_df2['curr_loss'] = record_df2['loss_rpn_cls'] + record_df2['loss_rpn_regr'] + record_df2['loss_class_cls'] + \
                              record_df2['loss_class_regr']

    r_epochs1 = len(record_df1)
    r_epochs2 = len(record_df2)

    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss Value', fontsize=12)

    plt.plot(np.arange(0, r_epochs1), record_df1['curr_loss'], ':o', color='r')
    plt.plot(np.arange(0, r_epochs1), record_df1['curr_loss_val'], ':o', color='b')
    plt.plot(np.arange(0, r_epochs2), record_df2['curr_loss'], '-o', color='r')
    plt.plot(np.arange(0, r_epochs2), record_df2['curr_loss_val'], '-o', color='b')

    yloc = [-25, 33, 31, 60, 37, 60, 37]
    for i in range(0, r_epochs2):
        ax.annotate('LR={:.1e}'.format(calculate_learning_rate(i, mode='adaptive')), xy=(i, record_df2['curr_loss'][i]),
                    xycoords='data',
                    bbox=dict(boxstyle="round", fc="white", ec="gray"),
                    xytext=(0, yloc[i]),
                    textcoords='offset points', ha='center', size=10,
                    arrowprops=dict(arrowstyle="->"))
    for i in range(0, r_epochs1):
        ax.annotate('LR=1.0e-5', xy=(i, record_df1['curr_loss'][i]),
                    bbox=dict(boxstyle="round", fc="white", ec="gray"),
                    xytext=(2, 1.4), ha='center', size=10,
                    arrowprops=dict(arrowstyle="->"))

    plt.grid()
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xticks(np.arange(0, max(r_epochs1, r_epochs2), step=1),
               labels=map(str, range(1, max(r_epochs1, r_epochs2) + 1)))
    ax.set_aspect(1.0 / ax.get_data_ratio() * 0.45)
    plt.legend(['training, constant LR', 'validation, constant LR', 'training, varying LR', 'validation, varying LR'],
               prop={'size': 10})
    plt.savefig(f'models/logs/learning.eps', format='eps')
    plt.show()


# train(im_size=600, train_mode=1, lr_mode='adaptive')
# train(im_size=600, train_mode=1, lr_mode='constant')
# plot_training_comp('400','400_adaptive')
