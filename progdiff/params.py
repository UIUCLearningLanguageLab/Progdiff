"""
use only dictionaries to store parameters.
ludwig works on dictionaries and any custom class would force potentially unwanted logic on user.
using non-standard classes here would also make it harder for user to understand.
any custom classes for parameters should be implemented by user in main job function only.
keep interface between user and ludwig as simple as possible
"""
from dataclasses import dataclass, fields
from typing import Tuple, Optional

# will submit 3*2=6 jobs, each using a different learning rate and "configuration"
param2requests = {
}

param2default = {
    'corpus_path': 'progdiff/dataset/raw_childes.csv',
    'category_file_path': 'progdiff/dataset/categories.csv',
    'vocab_size': 1024,
    'window_size': None,

    'device': 'cpu',
    'embedding_size': 16,
    'hidden_layer_info_list': (("lstm", 256),),
    'weight_init': 0.0001,

    'criterion': 'cross_entropy',

    'num_epochs': 1,
    'optimizer': 'adagrad',
    'learning_rate': 0.1,

    'evaluation_layer': 0,
    'sequence_list': None,

    'test_frequency': 1,
}
#
# param2default_corpus = {
#     'vocab_size': ,
#     'target_list': ,
#     'window_size': ,
# }
#
# param2default_model = {
#
#
#      ,

#
#    : ,
#
#
#
#     'category_file_path': ,
#     'test_frequency': ,
#     'evaluation_layer: '
# }
#
# param2default = {
#     'composition_fn': 'native',
# }
# for k in param2default_corpus:
#     assert k not in param2default_model
# param2default.update(param2default_corpus)
# param2default.update(param2default_model)
#
# @dataclass
# class CorpusParams:
#     vocab_size: int
#     target_list: list
#     window_size: int
#
#     @classmethod
#     def from_param2val(cls, param2val):
#         field_names = set(f.name for f in fields(cls))
#         return cls(**{k: v for k, v in param2val.items() if k in field_names})
#
# @dataclass
# class ModelParams:
#     embedding_size: int
#     hidden_layer_info_list: list
#     weight_init: float
#     device: str
#     learning_rate:float
#     num_epochs: int
#
#     category_file_path: str
#     test_frequency: int
#     evaluation_layer: int
#
#     @classmethod
#     def from_param2val(cls, param2val):
#         field_names = set(f.name for f in fields(cls))
#         return cls(**{k: v for k, v in param2val.items() if k in field_names})
#
# @dataclass
# class Params:
#     model_params: ModelParams
#     composition_fn: str
#     corpus_params: CorpusParams
#
#     @classmethod
#     def from_param2val(cls, param2val):
#
#         # exclude keys from param2val which are added by Ludwig.
#         # they are relevant to job submission only.
#         tmp = {k: v for k, v in param2val.items()
#                if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
#
#         if param2val['dsm'] == 'count':
#             dsm_params = WWParams.from_param2val(tmp)
#         elif param2val['dsm'] == 'random':
#             dsm_params = WDParams.from_param2val(tmp)
#         elif param2val['dsm'] == 'w2v':
#             dsm_params = Word2VecParams.from_param2val(tmp)
#         elif param2val['dsm'] == 'srn':
#             dsm_params = RNNParams.from_param2val(tmp)
#         elif param2val['dsm'] == 'lstm':
#             dsm_params = LSTMParams.from_param2val(tmp)
#         elif param2val['dsm'] == 'gpt':
#             dsm_params = GPTParams.from_param2val(tmp)
#         else:
#             raise AttributeError(f'Invalid arg to "dsm" "{param2val["dsm"]}".')
#         if param2val['corpus'] == 'AyB':
#             corpus_params = AyBCorpusParams.from_param2val(tmp)
#         elif param2val['corpus'] == 'Childes':
#             corpus_params = ChildesCorpusParams.from_param2val(tmp)
#         return cls(dsm=param2val['dsm'],
#                    corpus=param2val['corpus'],
#                    composition_fn=param2val['composition_fn'],
#                    corpus_params=corpus_params,
#                    dsm_params=dsm_params,
#                    )
#
#
