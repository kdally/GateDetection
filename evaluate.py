import warnings
warnings.filterwarnings("ignore")
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter
from shapely import geometry
import test


def c2pol(list_pt):
    """ Convert list of corners to a Polygon object """

    pol = geometry.Polygon([(list_pt[0], list_pt[1]),
                            (list_pt[2], list_pt[3]),
                            (list_pt[4], list_pt[5]),
                            (list_pt[6], list_pt[7])])
    return pol


def read_coordinate_csv(name):
    """ Read CSV file with all gate corners, convert them to a list of list """

    with open(name, 'r') as f:
        reader = csv.reader(f)
        combined_list = [[row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4]),
                          int(row[5]), int(row[6]), int(row[7]), int(row[8])] for row in reader]
        return combined_list


def associate_gates(predicteds, candidates):
    """ For a given list of predicted gates and candidate (reference) gates, associate them by pair
        If lists vary in length, the missing partner is 'None'
    """

    # if there are more reference candidate gates than predictions, each prediction is easily matched with the best candidate.
    # The candidate gates without partners are matched with 'None'
    if len(predicteds) < len(candidates):
        associated = associate_lists(predicteds, candidates)

        candidate_matches = list(map(itemgetter(1), associated))

        # find which candidate gate was not matched with any prediction (so the missed gates)
        missing_gates = (gate for gate in candidates if gate not in candidate_matches)
        for missed_gate in missing_gates:
            associated.append(['None', missed_gate])

    # if there are more predictions than candidate gates, each candidate is matched with the best prediction.
    # The predictions  without partners are matched with 'None'
    elif len(predicteds) > len(candidates):
        associated = associate_lists(candidates, predicteds, mode='switched')

        candidate_matches = list(map(itemgetter(1), associated))

        # find which prediction was not matched with any candidate gate (so the duplicate predictions)
        duplicate_gates = (gate for gate in predicteds if gate not in candidate_matches)
        for duplicate_gate in duplicate_gates:
            associated.append([duplicate_gate, 'None'])

    # if the amount of predicted and candidate gates is the same
    else:
        associated = associate_lists(predicteds, candidates)

    return associated


def associate_lists(source_list, target_list, mode='normal'):
    """ Associate two lists of gates together by best matching each gate couple by area (provided they intersect)
        If the lists are not of the same length, the 'extra' gates are not matched
        (i.e. duplciate prediction or missed candidates)

       Args:
        source_list: list of gates of shorter length (predicted gates in normal mode)
        target_list: list of gates of longer length (candidate gates in normal mode)

       Returns:
        associated_list: list of matched gate couples
    """

    associated_list = []

    for source_gate in source_list:
        area_diff = []
        dist = []
        for target_gate in target_list:
            area_diff.append([abs(source_gate.area - target_gate.area)])
            dist.append(source_gate.centroid.distance(target_gate.centroid))

        # order marches by area difference
        target_match = target_list[area_diff.index(min(area_diff))]

        # if they do not intersect, the best match is the one with the lowest distance to its centroid
        if not target_match.intersects(source_gate):
            target_match = target_list[dist.index(min(dist))]

        # when there are more candidates than predictions
        if mode == 'normal':
            associated_list.append([source_gate, target_match])

        # when there are more predictions than candidates
        else:
            associated_list.append([target_match, source_gate])

    return associated_list


def evaluate(im_size, mode='testing', iou_TP_threshold=0.6, detect_threshold=0.99, overlap_threshold=0.6):
    """ Evaluation of the model for a given image size and thresholds
    Args:
        im_size: trained model available for 300, 400 or 600. Input images are resized to this size.
        mode: 'testing' means new predictions will be obtained. Change if they have already been generated.
        iou_TP_threshold: minimum IoU with true inner gate corners for proposal to be counted as True Positive (TP)
        detect_threshold: minimum class belonging probability for a proposal to be accepted
        overlap_threshold: maximum IoU between two proposals

    Returns:
        TP_r: True Positive Ratio
        FP_r: False Positive Ratio
    """

    if mode == 'testing':
        test.test(im_size=im_size, mode='evaluation',
                  detect_threshold=detect_threshold, overlap_threshold=overlap_threshold)

    prediction_rows = read_coordinate_csv(f'output/predictions/frcnn_size{int(im_size)}_p{int(detect_threshold * 100)}/'
                                          f'predicted_corners_size{int(im_size)}_p{int(detect_threshold * 100)}.csv')

    prediction_names = list(map(itemgetter(0), prediction_rows))

    # list of prediction images without duplicates
    prediction_names_unique = list(OrderedDict.fromkeys(prediction_names))

    reference_rows = read_coordinate_csv('data/original/corners.csv')
    reference_names = list(map(itemgetter(0), reference_rows))

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for pred_image in prediction_names_unique:

        # true gates in the image from reference data
        candidates_gates = [c2pol(reference_rows[i][1:9]) for i, ref_image in enumerate(reference_names)
                            if ref_image == pred_image]

        # predicted gates in the image
        predicted_gates = [c2pol(prediction_rows[j][1:9]) for j, any_pred_image in enumerate(prediction_names)
                           if any_pred_image == pred_image]

        # each true gate has been associated with a predicted gate, or to 'None' (if one list is longer than the other)
        # each gate in a given column appears only once in the array
        associated_gates = associate_gates(predicted_gates, candidates_gates)

        for couple in associated_gates:

            # when a prediction was matched with an actual gate
            if couple[0] != 'None' and couple[1] != 'None':
                intersection = couple[0].intersection(couple[1]).area
                union = couple[0].union(couple[1]).area
                iou = intersection / union

                if iou >= iou_TP_threshold:
                    TP += 1

                # if the prediction did not have a high enough IoU
                else:
                    FN += 1

                # # visualization
                # if iou < iou_TP_threshold:
                #     x1, y1 = couple[0].exterior.xy
                #     plt.plot(x1, y1)
                #
                #     x2, y2 = couple[1].exterior.xy
                #     plt.plot(x2, y2)
                #     plt.xlim(0, 360)
                #     plt.ylim(0, 360)
                #     print(iou)
                #     plt.show()

            # If a true gate was not assigned any prediction
            elif couple[0] == 'None':
                FN += 1

            # If a prediction was not assigned any true gate
            elif couple[1] == 'None':
                FP += 1

        union_pol = predicted_gates[0]
        area = []
        for predicted_gate in predicted_gates:
            area.append(predicted_gate.area)
            union_pol = union_pol.union(predicted_gate)
        non_predicted_area = 360 * 360 - union_pol.union(union_pol).area

        # calculate the number of true negatives by seeing how many gates fit in the unpredicted area
        average_gate_area = sum(area) / len(area)
        TN += non_predicted_area / average_gate_area

    TP_r = TP / (TP + FN)
    FP_r = FP / (FP + TN)

    return [TP_r, FP_r]


def roc_curve_iou(iou_TP_threshold_range, im_size=600, overlap_threshold=0.6):
    """ Generate an ROC curve for different iou_TP_thresholds
    Args:
        iou_TP_threshold_range: range of IoU with true inner gate corners for proposal to be counted as True Positive (TP)
        im_size: trained model available for 300, 400 or 600. Input images are resized to this size.
        overlap_threshold: maximum IoU between two proposals

    Returns:
        None - (ROC curve saved automatically)
    """

    fig = plt.figure()
    ax = plt.gca()

    for i, iou_TP_threshold in enumerate(iou_TP_threshold_range, start=1):
        detect_threshold_range = np.array([0.99, 0.98, 0.95, 0.6])

        TP_r_list = [0.0]
        FP_r_list = [0.0]

        # for annotations
        x_loc = [38, 71, 79, 100]
        y_loc = [-45, -39, -20, -10]

        for j, detect_threshold in enumerate(detect_threshold_range, start=1):

            [TP_r, FP_r] = evaluate(im_size, 'testing', iou_TP_threshold, detect_threshold, overlap_threshold)
            TP_r_list.append(TP_r)
            FP_r_list.append(FP_r)
            if iou_TP_threshold == iou_TP_threshold_range[-1]:
                ax.annotate(f'p>{detect_threshold}', xy=(FP_r, TP_r), xycoords='data',
                            bbox=dict(boxstyle="round", fc="white", ec="gray"),
                            xytext=(x_loc[j-1], y_loc[j-1]),
                            textcoords='offset points', ha='center', size=11,
                            arrowprops=dict(arrowstyle="->"))
            print(
                f"Completed {int(((i-1) * len(detect_threshold_range) + j) / (len(iou_TP_threshold_range) * len(detect_threshold_range)) * 100)}% of evaluation")

        TP_r_list.append(1.0)
        FP_r_list.append(1.0)
        plt.plot(FP_r_list, TP_r_list, '-ok', linewidth=2, c=np.random.rand(3, ))

    plt.xlabel('False Positive Ratio', fontsize=11)
    plt.ylabel('True Positive Ratio', fontsize=11)
    plt.legend([f'IOU_th={iou_TP_threshold}' for iou_TP_threshold in iou_TP_threshold_range], prop={'size': 11})
    plt.grid()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_aspect(1.0 / ax.get_data_ratio() * 0.45)
    plt.savefig(f'output/evaluation/roc_curve_iou_size{int(im_size)}.eps', format='eps')
    plt.show()
    print(f'Output figure saved at output/evaluation/roc_curve_iou_size{int(im_size)}.eps')


def size_effect(im_size_range):
    """ Study the impact of image size on computational time

    Args:
        im_size_range: range of image sizes

    Returns:
        None - (CSV table saved automatically)
    """

    output_file = open('output/evaluation/im_size_performance.csv', 'w')
    writer = csv.DictWriter(
        output_file, fieldnames=["im_size", "time/img", 'TP_r', 'FP_r'])
    writer.writeheader()
    writer = csv.writer(output_file)
    iou_TP_threshold = 0.5

    # for samll images, relatively fewer gates are detected so the probability detection threshold is slightly lowered,
    # and maximum IoU overlap between boxes increased
    detect_thresholds = [0.5, 0.6, 0.6]
    overlap_thresholds = [0.6, 0.5, 0.5]

    for i, im_size in enumerate(im_size_range):
        avg_time = test.test(im_size=im_size, mode='evaluation',
                             detect_threshold=detect_thresholds[i], overlap_threshold=overlap_thresholds[i])
        [TP_r, FP_r] = evaluate(im_size, 'no_testing', iou_TP_threshold, detect_thresholds[i], overlap_thresholds[i])
        writer.writerow([im_size, avg_time, TP_r, FP_r])

    output_file.close()
    print(f'Output CSV file saved at output/evaluation/im_size_performance.csv')


# roc_curve_iou([0.5, 0.6, 0.7], im_size=600, overlap_threshold=0.5)
# roc_curve_iou([0.5, 0.6, 0.7], im_size=300, overlap_threshold=)
# roc_curve_iou([0.5, 0.6, 0.7], im_size=400, overlap_threshold=)
# size_effect([300, 400, 600])
# print(evaluate(im_size=600, iou_TP_threshold=0.6, detect_threshold=0.6, overlap_threshold=0.5))

