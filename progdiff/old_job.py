import pandas as pd
import time
from progdiff.distributional_models.tasks.cohyponym_task import CohyponymTask
from progdiff.distributional_models.datasets.childes import Childes
from progdiff.distributional_models.tasks.categories import Categories
from .progdiff import create_corpus
from progdiff.distributional_models.models.neural_network import NeuralNetwork
from progdiff.distributional_models.tasks.classifier import classify


def main(param2val):
    """This function is run by Ludwig on remote workers."""

    #  load the category file
    category_file_path = "input_data/categories.csv"
    # category_file_path = "/media/ludwig_data/Progdiff/input_data/categories.csv"
    the_categories = Categories()
    the_categories.create_from_category_file(category_file_path)

    # # load the corpus and create the vocab
    vocab_size = 4096
    childes_db_path = 'input_data/childes.csv'
    childes_pkl_path = "input_data/childes_ludwig.pkl"
    # childes_pkl_path = '/media/ludwig_data/Progdiff/input_data/childes_ludwig.pkl'
    # corpus = create_corpus(childes_db_path, vocab_size, target_list)
    corpus = Childes.load_from_file(childes_pkl_path)
    missing_words = corpus.create_vocab(vocab_size=param2val['vocab_size'], include_list=the_categories.instance_list,
                                        include_unknown=True)
    the_categories.remove_instances(missing_words)

    model = NeuralNetwork(param2val['embedding_size'],
                          param2val['hidden_layer_info_list'],
                          param2val['weight_init'],
                          corpus.vocab_index_dict,
                          param2val['criterion'],
                          device=param2val['device'])
    print(model.layer_list)

    cohyponym_task = CohyponymTask(the_categories, num_thresholds=21)

    performance = {}
    for i in range(param2val['num_epochs']):
        for j in range(corpus.num_documents):
            doc_sequence_list = corpus.flatten_corpus_lists(corpus.document_list[j])
            corpus.x_list, corpus.y_list = corpus.create_sequence_list(doc_sequence_list,
                                                                       corpus.vocab_index_dict,
                                                                       corpus.unknown_token,
                                                                       window_size=param2val['window_size'])
            lm_took, perplexity = model.train_sequence(corpus.x_list, corpus.y_list, param2val['optimizer'], param2val['learning_rate'])

            if j % param2val['test_frequency'] == 0:
                weight_matrix = model.get_weights(param2val['evaluation_layer'])

                start_time = time.time()
                the_categories.set_instance_feature_matrix(weight_matrix, corpus.vocab_index_dict)
                the_categories.create_xy_lists()
                mean_ba = cohyponym_task.run_cohyponym_task()
                ba_took = time.time() - start_time
                start_time = time.time()
                train, test = classify(the_categories, param2val['classifier_hidden_size'], test_proportion=param2val['test_proportion'],
                                       num_epochs=param2val['classifier_epochs'], learning_rate=param2val['classifier_lr'])

                train_acc = train['correct'].mean()
                test_acc = test['correct'].mean()
                classify_took = time.time() - start_time
                print(
                    f"doc: {i}-{len(doc_sequence_list)}   took: {lm_took:0.2f}/{ba_took:0.2f}/{classify_took:0.2f}   perp: {perplexity:0.1f}  ba:{mean_ba:0.3f}    train:{train_acc:0.3f}   test:{test_acc:0.3f}")
            #
            # mean_ba = cohyponym_task.run_cohyponym_task()
            #     print(f"doc: {j}   took: {took:0.2f}   perp: {perplexity:0.2f}   ba: {mean_ba:0.2f}")

            index = f"{i}-{j}"
            performance[index] = lm_took

    series_list = []
    for k, v in performance.items():
        s = pd.Series(v, index=k)
        s.name = 'took'
        series_list.append(s)
    return series_list
