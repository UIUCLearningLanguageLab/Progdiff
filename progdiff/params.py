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
    'random_seed': None,
    'device': 'mps',

    'corpus_path': 'input_data/childes',
    'category_file_path': 'input_data/categories.csv',
    'num_document_batches': 16,
    'document_order': 'age_ordered',
    'vocab_size': 4096,

    # Model Params
    'model_type': 'lstm',
    'weight_init': 0.0001,
    'save_path': 'models/',
    'save_freq': 100,
    'sequence_length': 8,

    # SRN & LSTM Params
    'rnn_embedding_size': 0,
    'rnn_hidden_size': 512,

    # W2V Params
    'w2v_embedding_size': 0,
    'w2v_hidden_size': 256,
    'corpus_window_size': 16,

    # Transformer params
    'transformer_embedding_size': 64,
    'transformer_num_heads': 8,
    'transformer_attention_size': 32,
    # 'transformer_num_layers': 3,
    'transformer_hidden_size': 128,
    'transformer_target_output': 'single_y',

    # Training Params
    'num_epochs': 5,
    'criterion': 'cross_entropy',
    'optimizer': 'adagrad',
    'learning_rate': 0.01,
    'batch_size': 64,
    'dropout_rate': 0.0,
    'l1_lambda': 0.0,
    'weight_decay': 0.0,

    # evaluation params
    'eval_freq': 1,
    'evaluation_layer': 'output',
    'sequence_list': None,

    # cohyponym task params
    'run_cohyponym_task': True,
    'cohyponym_similarity_metric': 'correlation',
    'cohyponym_num_thresholds': 51,
    'cohyponym_only_best_thresholds': True,

    # classifier task params
    'run_classifier_task': False,
    'num_classifiers': 1,
    'classifier_hidden_sizes': (),
    'classifier_num_folds': 10,
    'classifier_num_epochs': 10,
    'classifier_learning_rate': .05,
    'classifier_batch_size': 64,
    'classifier_criterion': 'cross_entropy',
    'classifier_optimizer': 'adam',
    'classifier_device': 'mps',

    # generate sequence task params
    'generate_sequence': True,
    'prime_token_list': ('look', 'at'),
    'generate_sequence_length': 10,
    'generate_temperature': 1.0
}
