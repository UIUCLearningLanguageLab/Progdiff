"""
use only dictionaries to store parameters.
ludwig works on dictionaries and any custom class would force potentially unwanted logic on user.
using non-standard classes here would also make it harder for user to understand.
any custom classes for parameters should be implemented by user in main job function only.
keep interface between user and ludwig as simple as possible
"""

# will submit 3*2=6 jobs, each using a different learning rate and "configuration"
param2requests = {
}

param2default = {
    'corpus_path': 'input_data/childes',
    'category_file_path': 'input_data/categories.csv',
    'vocab_size': 4096,
    'window_size': None,

    'device': 'mps',
    'embedding_size': 64,
    'hidden_layer_info_list': (('lstm', 256),),
    'weight_init': 0.0001,
    'sequence_length': 3,

    'criterion': 'cross_entropy',

    'num_epochs': 10,
    'optimizer': 'adagrad',
    'learning_rate': 0.01,
    'batch_size': 1,

    'evaluation_layer': 0,
    'sequence_list': None,

    'eval_freq': 10,

    'run_cohyponym_task': True,
    'num_thresholds': 31,

    'num_classifiers': 10,
    'run_classifier_task': True,
    'classifier_hidden_sizes': [],
    'test_proportion': .1,
    'classifier_epochs': 1,
    'classifier_lr': .2
}