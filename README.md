Phase 1: Classical approach using feature matching to get homography, and poisson image blending.

Phase 2: HomographyNet, with both supervised and unsupervised models, to compute homography, then blend

1.The code for this implementation is located in Wrapper.py. To work with the training or testing dataset, update the DIR_PATH variable accordingly.

For the training set:
DIR_PATH = f'/Phase1/Data/Train/'

For the testing set:
DIR_PATH = f'Phase1/Data/Test/'

2.Specify the desired subset for training and testing as follows. For example, to use Set1:
path = "Set1"

3. Please run the file.
