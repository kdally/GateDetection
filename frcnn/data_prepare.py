import warnings
warnings.filterwarnings("ignore")
import os, csv, random, shutil
import pandas as pd


def organize():
    """ Split the original dataset in a random fashion between training (90%) and testing (10%) data.
        Directories are reorganized accordingly.
    """

    path = './data/original'
    all_samples = os.listdir(f'{path}')
    all_samples.sort()

    if '.DS_Store' in all_samples:
        all_samples.remove('.DS_Store')

    all_samples.remove('corners.csv')
    sample_count = int(len(all_samples)/2)

    if sample_count != 308:
        print("Incorrect input data was passed. Please unload the full content of the 'WashingtonOBRace' folder including " \
        "'corner.csv', images and masks with names 'img_X.png' and 'mask_X.png' in the data/original folder.")
        exit()

    try:
        os.stat(f'./data/testing')
        shutil.rmtree(f'./data/testing')
    except:
        pass
    try:
        os.stat(f'./data/training')
        shutil.rmtree(f'./data/training')
    except:
        pass

    os.makedirs(f'data/testing/images')
    os.makedirs(f'data/testing/masks')
    os.makedirs(f'data/training/images/gate')
    os.makedirs(f'data/training/masks/gate')

    idx = range(0, sample_count)
    testing_idx = random.sample(idx, round(0.1 * sample_count))

    for i in idx:

        source_im = f'{path}/{all_samples[i]}'
        source_mask = f'{path}/{all_samples[sample_count+i]}'

        if i in testing_idx:
            destination = 'data/testing/images/'
            shutil.copy(source_im, destination)

            destination = 'data/testing/masks/'
            shutil.copy(source_mask, destination)

        else:
            destination = 'data/training/images/gate/'
            shutil.copy(source_im, destination)

            destination = 'data/training/masks/gate/'
            shutil.copy(source_mask, destination)

    return


def reformat(scale=0.2):
    """ Read the original SCV corner file, extract its information for training data and rescale the bounding boxes
        to match the outer gate frame.
    """

    input_file = open(f'./data/original/corners.csv', 'r')

    training_im_list = os.listdir('./data/training/images/gate')
    training_im_list.sort()

    list_of_rows = list(csv.reader(input_file))

    output_list = []

    for row in list_of_rows:

        if row[0] in training_im_list:
            path = f'data/training/images/gate/{row[0]}'

            x_min = min(int(row[1]), int(row[7]))
            y_min = min(int(row[2]), int(row[4]))
            x_max = max(int(row[3]), int(row[5]))
            y_max = max(int(row[6]), int(row[8]))
            class_name = 'gate'

            # obtain the outer frame of the gate (ratio of 0.2 on average w.r.t. inner frame size)
            w = (x_max - x_min)
            h = (y_max - y_min)
            x_min -= w * scale
            x_max += w * scale
            y_min -= h * scale
            y_max += h * scale

            output_list.append(path + ',' + str(int(x_min)) + ',' + str(int(y_min)) + ',' + str(int(x_max)) +
                               ',' + str(int(y_max)) + ',' + class_name)

    output_data = pd.DataFrame()
    output_data['format'] = output_list
    output_data.to_csv('data/training/corners_training.txt', header=False, index=False, sep=' ')

