# data files
raw_data_file = 'combined.txt'
data_file = 'final_data2.pkl'
dict_file = 'dict_data2.pkl'

# dataset size
train_ratio = 0.9  # the percentage of samples to be training data
train_test_split_seed = 17  # how to split the training / testing data
title_len_filter = 1 # remove all the data with title length <= title_len_filter
title_len_upper = 200 # remove all data with title length > title_len_upper
text_len_filter = 1 # remove all the data with content length <= content_len_filter

# configuration
using_boostrap = True   # ensure there are equal amount of positive and negative samples
using_unknown_symbol = True  # replace rare symbols with <unk>
word_occur_threshold = 50
using_subset = 0 #using a subset of testing/training data for fast validation
                 # 0: evaluation will be performed on the whole testing set

# training config
decay_rate = 0.1
decay_round = 3
opt_type = 'adam' # 'grad' for stochastic gradient descent
                  # 'adam' for Adam

# default config values
using_nonlinear = 'True'    # tanh nonlinear layer before softmax
using_dropout = 'False'
dropout_rate = 0.5
using_l2_reg = 'False'
l2reg_coef = 1e-6
using_convolution = 'True'   # convolution layer and params
filter_size = 3
feature_size = 50
cpu_name = '/cpu:0'
gpu_name = '/gpu:2'
gpu_usage_upper_bound = 0.8  # upper bound for GPU memory used by TF
using_bidirect = 'False'

# model parameters
n_embed = 100         # when using pretrain, make sure n_embed = pretrain_dim
n_hidden = n_embed
n_class = 2
n_final = 30          # output dimension for the non-linear tanh layer when using_nonlinear == True
batch_size = 128
num_steps = 150       # sequence length for content
num_steps_title = 150  # sequence length for title
init_scale = 0.1      # intialization scale

# pretrain info
pretrain_file = 'pretrain/pretrain_embedding_100.pkl'
pretrain_dim = 100
pretrain_vocab_size = 7000 # Note: see below
# when using_pretrian == True, make sure pretrain_vocab_size >= actual pretrian dictionary size
# unless the chars in the pretrain file are sorted according to their frequencies
