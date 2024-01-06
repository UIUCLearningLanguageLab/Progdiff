import time
from distributional_models.corpora.childes import Childes
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.cohyponym_task import CohyponymTask
from distributional_models.tasks.classifier import classify
from distributional_models.models.neural_network import NeuralNetwork


def main():
    import params
    progdiff(params.param2default, "local")


def progdiff(param2val, run_location):

    if run_location == 'local':
        param2val['corpus_path'] = "../" + param2val['corpus_path']  # "input_data/childes"
        param2val['category_file_path'] = "../" + param2val['category_file_path']  # "input_data/categories"
    elif run_location == 'ludwig_local':
        pass
    elif run_location == 'ludwig_cluster':
        param2val['corpus_path'] = "/media/ludwig_data/Progdiff/" + param2val['corpus_path']
        param2val['category_file_path'] = "/media/ludwig_data/Progdiff/" + param2val['category_file_path']
    else:
        raise ValueError(f"Unrecognized run location {run_location}")

    the_categories = init_categories(param2val['category_file_path'])
    the_corpus, missing_words = init_corpus(param2val['vocab_size'],
                                            the_categories.instance_list,
                                            param2val['corpus_path'])
    the_categories.remove_instances(missing_words)
    the_model = init_model(the_corpus,
                           param2val['embedding_size'],
                           param2val['hidden_layer_info_list'],
                           param2val['weight_init'],
                           param2val['device'],
                           param2val['criterion'])

    performance_dict = train_model(the_corpus, the_model, the_categories, param2val)

    return performance_dict


def init_categories(category_file_path):
    the_categories = Categories()
    the_categories.create_from_category_file(category_file_path)
    return the_categories


def init_corpus(vocab_size, include_list, corpus_path):
    # TODO this can be improved to catch specific exceptions, like is the file there, and the error that will occur
    # if you loaded a childes instance created under a different path
    try:
        the_corpus = Childes.load_from_file(corpus_path+'.pkl')
    except:
        the_corpus = create_corpus(corpus_path+".csv")
        the_corpus.save_to_pkl_file(corpus_path)
    missing_words = the_corpus.create_vocab(vocab_size=vocab_size, include_list=include_list, include_unknown=True)
    return the_corpus, missing_words


def create_corpus(corpus_path, language="eng", collection_name=None, age_range_tuple=(0, 1000),
                  sex_list=None, add_punctuation=True, exclude_target_child=True, num_documents=0):
    the_corpus = Childes()
    the_corpus.get_documents_from_childes_db_file(input_path=corpus_path,
                                                  language=language,
                                                  collection_name=collection_name,
                                                  age_range_tuple=age_range_tuple,
                                                  sex_list=sex_list,
                                                  add_punctuation=add_punctuation,
                                                  exclude_target_child=exclude_target_child,
                                                  num_documents=num_documents)
    return the_corpus


def init_model(corpus, embedding_size, hidden_layer_info_list, weight_init, device, criterion):
    model = NeuralNetwork(corpus, embedding_size, hidden_layer_info_list, weight_init, criterion, device=device)
    return model


def cohyponym_task(the_categories, num_thresholds):
    start_time = time.time()
    the_cohyponym_task = CohyponymTask(the_categories, num_thresholds=num_thresholds)
    mean_ba = the_cohyponym_task.run_cohyponym_task()
    took = time.time() - start_time
    return the_cohyponym_task, mean_ba, took


def classifier_task(the_categories, classifier_hidden_sizes, test_proportion, classifier_epochs,
                    classifier_lr):
    start_time = time.time()
    train_df, test_df = classify(the_categories, classifier_hidden_sizes, test_proportion=test_proportion,
                                 num_epochs=classifier_epochs, learning_rate=classifier_lr)
    train_acc = train_df['correct'].mean()
    test_acc = test_df['correct'].mean()
    took = time.time() - start_time
    return train_df, test_df, train_acc, test_acc, took


def train_model(corpus, model, the_categories, train_params):

    performance_dict = {}

    for i in range(train_params['num_epochs']):
        took_sum = 0
        tokens = 0
        perplexity_sum = 0
        n = 0
        for j in range(len(corpus.document_list)):

            doc_sequence_list = corpus.flatten_corpus_lists(corpus.document_list[j])
            corpus.x_list, corpus.y_list = corpus.create_sequence_list(doc_sequence_list,
                                                                       corpus.vocab_index_dict,
                                                                       corpus.unknown_token,
                                                                       window_size=train_params['window_size'])
            took, perplexity = model.train_sequence(corpus.x_list,
                                                    corpus.y_list,
                                                    train_params['optimizer'],
                                                    train_params['learning_rate'],
                                                    batch_size=train_params['batch_size'],
                                                    sequence_length=train_params['sequence_length'])
            tokens += len(doc_sequence_list)
            took_sum += took
            perplexity_sum += perplexity*len(doc_sequence_list)
            n += 1

            if j % train_params['eval_freq'] == 0:
                # model training output
                perplexity_mean = perplexity_sum/tokens
                output_string = f"{i}-{j}-{tokens:<9}  took:{took_sum:2.2f}  perp:{perplexity_mean:<7.2f}"
                took_sum = 0
                tokens = 0
                perplexity_sum = 0
                n = 0

                weight_matrix = model.get_weights(train_params['evaluation_layer'])
                the_categories.set_instance_feature_matrix(weight_matrix, corpus.vocab_index_dict)

                if train_params['run_cohyponym_task']:
                    the_cohyponym_task, mean_ba, ba_took = cohyponym_task(the_categories,
                                                                          train_params['num_thresholds'])
                    output_string += f"  BA:{mean_ba:0.3f}"

                if train_params['run_classifier_task']:

                    train_sum = 0
                    test_sum = 0

                    for k in range(train_params['num_classifiers']):
                        the_categories.create_xy_lists()
                        train_df, \
                            test_df, \
                            train_acc, \
                            test_acc, \
                            classify_took = classifier_task(the_categories,
                                                            train_params['classifier_hidden_sizes'],
                                                            train_params['test_proportion'],
                                                            train_params['classifier_epochs'],
                                                            train_params['classifier_lr'])
                        train_sum += train_acc
                        test_sum += test_acc

                    if train_params['num_classifiers'] > 0:
                        train_mean = train_sum / train_params['num_classifiers']
                        test_mean = test_sum / train_params['num_classifiers']
                        output_string += f"  Classify:{train_mean:0.3f}-{test_mean:0.3f}"

                print(output_string)
    return performance_dict


if __name__ == "__main__":
    main()
