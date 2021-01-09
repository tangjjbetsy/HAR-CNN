## FILE DICTIONARIES
# LOCAL
TRAINFOLD = "../data/train/Inertial Signals/"
TESTFOLD = "../data/test/Inertial Signals/"
TRAINFEATURE = ["../data/train/X_train.txt", "../data/train/y_train.txt"]
TESTFEATURE = ["../data/test/X_test.txt", "../data/test/y_test.txt"]

# # SERVER
# TRAINFOLD = "/data/419yangnan/datasets/har_data/data/train/Inertial Signals/"
# TESTFOLD = "/data/419yangnan/datasets/har_data/data/test/Inertial Signals/"
# TRAINFEATURE = ["/data/419yangnan/datasets/har_data/data/train/X_train.txt", "/data/419yangnan/datasets/har_data/data/train/y_train.txt"]
# TESTFEATURE = ["/data/419yangnan/datasets/har_data/data/test/X_test.txt", "/data/419yangnan/datasets/har_data/data/test/y_test.txt"]

TRAINLIST = ["body_acc_x_train.txt",
            "body_acc_y_train.txt",
            "body_acc_z_train.txt",
            "body_gyro_x_train.txt",
            "body_gyro_z_train.txt",
            "body_gyro_y_train.txt",
            "total_acc_x_train.txt",
            "total_acc_y_train.txt",
            "total_acc_z_train.txt"
            ]
TESTLIST = ["body_acc_x_test.txt",
            "body_acc_y_test.txt",
            "body_acc_z_test.txt",
            "body_gyro_x_test.txt",
            "body_gyro_z_test.txt",
            "body_gyro_y_test.txt",
            "total_acc_x_test.txt",
            "total_acc_y_test.txt",
            "total_acc_z_test.txt"
            ]
PAIR = {1:[1,2], 
        2:[2,1],
        3:[1,3],
        4:[3,1],}
        # 5:[2,3],
        # 6:[3,2],
        # 7:[1,5],
        # 8:[5,1],
        # 9:[4,5],
        # 10:[5,4],
        # 11:[4,6],
        # 12:[6,4],
        # 13:[2,5],
        # 14:[5,2],
        # 15:[3,5],
        # 16:[5,3]
        
        
SAVEPATH = "save_models/"
DATE = "2020.5.14/"

# DATA SETTING
WINDOW_WIDTH = 128 # MAX_WINDOW_WIDTH = 128
NUM_CLASSES = 6
NUM_FEATURES_USED = 2
SPLIT_RATE = .95
TARGET_NAMES = {6:["Walking", 
                   "Walking Upstairs", 
                   "Walking Downstairs", 
                   "Sitting", 
                   "Standing", 
                   "Laying"],
                16:["W to U",
                    "U to W",
                    "W to D",
                    "D to W",
                    "U to D",
                    "D to U",
                    "W to S",
                    "S to W",
                    "Si to St",
                    "St to Si",
                    "S to L",
                    "L to S",
                    "U to S",
                    "S to U",
                    "D to S",
                    "S to D"],
                4:["Walking to Upstairs",
                   "Upstairs to Walking",
                   "Walking to Downstairs",
                   "Downstairs to Walking" ]}

# NETWORK CONSTANTS
AE1_DIM = 80
AE2_DIM = 5

# TRAINING CONSTANTS
EPOCH_NUM = 10
DISPLAY_NUM = 100
BATCH_SIZE = 64
SAVE_EVERY = 20
DEV_EVERY = 20
LOG_EVERY = 100
DATA_TYPE = 'trans'


