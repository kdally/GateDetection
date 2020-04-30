import warnings
warnings.filterwarnings("ignore")
import os
from frcnn import data_prepare
import train
import evaluate
import warnings
warnings.filterwarnings("ignore")

def main():
    print(
        "Welcome to the Automatic Gate Detection Algorithm. "
        "Make sure the entire dataset has been copied to  the folder /data/original")
    print('Press 1 to train the model, 2 to test the already-trained model.')
    mode = int(input('Choice: '))
    while mode != 1 and mode != 2:
        print('Wrong entry. Choose again.')
        mode = int(input('Choice: '))
    print('')

    path = './data/training/images/gate'
    if not os.path.exists(path) or len(os.listdir(path)) < 200:
        data_prepare.organize()
        data_prepare.reformat(scale=0.2)
        print('The dataset has been split between training and testing samples.')
        print('')

    # training mode
    if mode == 1:
        print("Enter image_size to be used by the model. Pre-trained models available for image_size = 300, 400, 600.")
        im_size = int(input("Size: "))
        print('')
        if im_size in [300, 400, 600]:
            print("Enter 1 to train a new model with generic pre-trained weights on the BCCD dataset.")
            print("Enter 2 to resume training on the WashingtonOBRace dataset.")
            train_mode = int(input("Choice: "))
            while train_mode != 1 and train_mode != 2:
                print('Wrong entry. Choose again.')
                train_mode = int(input('Choice: '))
        else:
            train_mode = 1
        print('')

        train.train(im_size, train_mode, epoch_length=1000, num_epochs=10, val_steps=200, lr_mode='adaptive')

    # testing mode
    if mode == 2:

        print('Press 1 to generate predictions and calculate the TP and FP ratios.')
        print('Press 2 to generate an ROC curve.')
        print('Press 3 to compare the computational time for different image sizes.')
        test_mode = int(input('Choice: '))
        while test_mode != 1 and test_mode != 2 and test_mode != 3:
            print('Wrong entry. Choose again.')
            test_mode = int(input('Choice: '))

        if test_mode == 1 or test_mode == 2:
            print('Choose image size. Trained models available for image_size = 300, 400, 600.')
            im_size = int(input("Size: "))
            while not os.path.isfile(f'models/model_frcnn_{im_size}.hdf5') and im_size not in [300, 400, 600]:
                print(
                    'Choose a different image size, no model available. Trained models available for image_size = 300, '
                    '400, 600.')
                im_size = int(input("Size: "))

            if test_mode == 1:
                TP_r, FP_r = evaluate.evaluate(im_size, mode='testing', iou_TP_threshold=0.6,
                                               detect_threshold=0.6, overlap_threshold=0.5)
                print(f'TP_r = {TP_r}, FP_r = {FP_r}')

            if test_mode == 2:
                evaluate.roc_curve_iou([0.5, 0.6, 0.7], im_size=im_size, overlap_threshold=0.5)

        elif test_mode == 3:
            evaluate.size_effect([300, 400, 600])

    print('')
    print("Task completed. Run again to perform another task.")


if __name__ == "__main__":
    main()
