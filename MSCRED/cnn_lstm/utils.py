# Parameter initialization

gap_time = 10  # gap time between each segment
win_size = [10, 30, 60]  # window size of each segment
step_max = 5 # maximum step of ConvLSTM

raw_data_path = '../data/synthetic_data_with_anomaly-s-1.csv'  # path to load raw data
model_path = '../MSCRED/'
train_data_path = "../data/train/"
test_data_path = "../data/test/"
reconstructed_data_path = "../data/reconstructed/"


train_start_id = 10
train_end_id = 800

test_start_id = 800
test_end_id = 2000

valid_start_id = 800
valid_end_id = 1000

training_iters = 5
save_model_step = 1

learning_rate = 0.0002

threhold = 0.005
alpha = 1.5