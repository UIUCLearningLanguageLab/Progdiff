import pandas as pd
from progdiff.distributional_models.tasks.cohyponym_task import CohyponymTask
from progdiff.distributional_models.datasets.childes import Childes
from .train_childes import create_corpus
from progdiff.distributional_models.models.neural_network import NeuralNetwork


def main(param2val):
    """This function is run by Ludwig on remote workers."""

    target_list, target_index_dict = CohyponymTask.load_category_file(param2val['category_file_path'])

    childes_pkl_path = "progdiff/dataset/childes_ludwig.pkl"

    # corpus = create_corpus(param2val['corpus_path'], param2val['vocab_size'], target_list)
    # corpus.save_to_pkl_file(childes_pkl_path)

    corpus = Childes.load_from_file(childes_pkl_path)

    corpus.create_vocab(vocab_size=param2val['vocab_size'], include_list=target_list, include_unknown=True)

    print(param2val['hidden_layer_info_list'])

    model = NeuralNetwork(param2val['embedding_size'],
                          param2val['hidden_layer_info_list'],
                          param2val['weight_init'],
                          corpus.vocab_index_dict,
                          param2val['criterion'],
                          device=param2val['device'])

    cohyponym_task = CohyponymTask(model,
                                   param2val['evaluation_layer'],
                                   param2val['category_file_path'],
                                   sequence_list=param2val['sequence_list'],
                                   num_thresholds=21)

    performance = {}
    for i in range(param2val['num_epochs']):
        for j in range(corpus.num_documents):
            doc_sequence_list = corpus.flatten_corpus_lists(corpus.document_list[j])
            corpus.x_list, corpus.y_list = corpus.create_sequence_list(doc_sequence_list,
                                                                       corpus.vocab_index_dict,
                                                                       corpus.unknown_token,
                                                                       window_size=param2val['window_size'])
            took, perplexity = model.train_sequence(corpus.x_list, corpus.y_list, param2val['criterion'],
                                                    param2val['optimizer'], param2val['learning_rate'])

            if j % param2val['test_frequency'] == 0:
                mean_ba = cohyponym_task.run_cohyponym_task()
                print(f"doc: {j}   took: {took:0.2f}   perp: {perplexity:0.2f}   ba: {mean_ba:0.2f}")

            index = f"{i}-{j}"
            performance[index] = took

    series_list = []
    for k, v in performance.items():
        s = pd.Series(v, index=k)
        s.name = 'took'
        series_list.append(s)
    return series_list
