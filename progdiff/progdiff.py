import time
import torch
from distributional_models.corpora.childes import Childes
from distributional_models.tasks.categories import Categories
from distributional_models.tasks.cohyponym_task import CohyponymTask
from distributional_models.tasks.classifier import classify
from distributional_models.models.lstm import SimpleLSTM
import torch.optim as optim


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
                           param2val['sequence_length'],
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


def init_model(corpus, block_size, embedding_size, hidden_layer_info_list, weight_init, device, criterion):
    model = SimpleLSTM(corpus,
                       block_size,
                       embedding_size,
                       hidden_layer_info_list,
                       weight_init,
                       criterion,
                       device=device)
    return model


def cohyponym_task(the_categories, num_thresholds):
    start_time = time.time()
    the_cohyponym_task = CohyponymTask(the_categories, num_thresholds=num_thresholds)
    mean_ba = the_cohyponym_task.run_cohyponym_task()
    took = time.time() - start_time
    return the_cohyponym_task, mean_ba, took


def classifier_task(the_categories, classifier_hidden_sizes, test_proportion, classifier_epochs,
                    classifier_lr, num_classifiers):

    start_time = time.time()

    train_0_sum = 0
    train_1_sum = 0
    train_final_sum = 0
    test_0_sum = 0
    test_1_sum = 0
    test_final_sum = 0

    for k in range(num_classifiers):
        the_categories.create_xy_lists()
        start_time = time.time()
        train_df_list, test_df_list = classify(the_categories, classifier_hidden_sizes, test_proportion=test_proportion,
                                               num_epochs=classifier_epochs, learning_rate=classifier_lr)
        train_acc_0 = train_df_list[0]['correct'].mean()
        train_acc_1 = train_df_list[1]['correct'].mean()
        train_acc_final = train_df_list[-1]['correct'].mean()
        test_acc_0 = test_df_list[0]['correct'].mean()
        test_acc_1 = test_df_list[1]['correct'].mean()
        test_acc_final = test_df_list[-1]['correct'].mean()

        train_0_sum += train_acc_0
        train_1_sum += train_acc_1
        train_final_sum += train_acc_final
        test_0_sum += test_acc_0
        test_1_sum += test_acc_1
        test_final_sum += test_acc_final

    train_0_mean = train_0_sum / num_classifiers
    train_1_mean = train_1_sum / num_classifiers
    train_final_mean = train_final_sum / num_classifiers
    test_0_mean = test_0_sum / num_classifiers
    test_1_mean = test_1_sum / num_classifiers
    test_final_mean = test_final_sum / num_classifiers

    took = time.time() - start_time

    return train_0_mean, train_1_mean, train_final_mean, test_0_mean, test_1_mean, test_final_mean, took


def prepare_batches(corpus, model, train_params):
    doc_index_list = corpus.flatten_corpus_lists(corpus.document_list[j])

    corpus.x_list, corpus.y_list = corpus.create_index_list(doc_index_list,
                                                            corpus.vocab_index_dict,
                                                            corpus.unknown_token,
                                                            window_size=train_params['window_size'])
    index_list = corpus.x_list + [corpus.y_list[-1]]

    sequence_list = corpus.create_sequence_lists(index_list, train_params['sequence_length'], 0)

    x_batches, y_batches = corpus.create_batches(sequence_list, train_params['batch_size'],
                                                 train_params['sequence_length'], 0)

    x_batches = [torch.tensor(x_batch, dtype=torch.long).to(model.device) for x_batch in x_batches]
    y_batches = [torch.tensor(y_batch, dtype=torch.long).to(model.device) for y_batch in y_batches]

    return x_batches, y_batches


def evaluate_model(i, j, model, the_categories, corpus, train_params, training_took, loss_sum, tokens_sum):
    loss_mean = loss_sum / tokens_sum

    output_string = f"{i}-{j}-{tokens_sum:<9}  loss:{loss_mean:<7.4f}"

    weight_matrix = model.get_weights(train_params['evaluation_layer'])
    the_categories.set_instance_feature_matrix(weight_matrix, corpus.vocab_index_dict)

    if train_params['run_cohyponym_task']:
        the_cohyponym_task, mean_ba, ba_took = cohyponym_task(the_categories,
                                                              train_params['num_thresholds'])
        output_string += f"  BA:{mean_ba:0.3f}"
    else:
        ba_took = 0

    if train_params['run_classifier_task']:
        train_0_mean, \
            train_1_mean, \
            train_final_mean, \
            test_0_mean, \
            test_1_mean, \
            test_final_mean, \
            classifier_took = classifier_task(the_categories,
                                              train_params['classifier_hidden_sizes'],
                                              train_params['test_proportion'],
                                              train_params['classifier_epochs'],
                                              train_params['classifier_lr'],
                                              train_params['num_classifiers'])
        output_string += f"  Classify0:{train_0_mean:0.3f}-{test_0_mean:0.3f}"
        output_string += f"  Classify1:{train_1_mean:0.3f}-{test_1_mean:0.3f}"
        output_string += f"  ClassifyN:{train_final_mean:0.3f}-{test_final_mean:0.3f}"

    else:
        classifier_took = 0

    output_string += f"  Took:{training_took:0.2f}-{ba_took:0.2f}-{classifier_took:0.2f}"
    print(output_string)


def train_model(corpus, model, the_categories, train_params):

    performance_dict = {}

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(train_params['num_epochs']):
        loss_sum = 0
        tokens_sum = 0

        for j in range(len(corpus.document_list)):

            start_time = time.time()
            x_batches, y_batches = prepare_batches(corpus, model, train_params)

            hidden = (torch.zeros(1, train_params['batch_size'], model.hidden_size).to(model.device),
                      torch.zeros(1, train_params['batch_size'], model.hidden_size).to(model.device))

            for x_batch, y_batch in zip(x_batches, y_batches):

                optimizer.zero_grad()
                output, hidden = model(x_batch, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())

                loss = criterion(output.view(-1, corpus.vocab_size), y_batch.view(-1))
                mask = y_batch.view(-1) != 0
                loss = (loss * mask).mean()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                tokens_sum += train_params['batch_size']

            training_took = time.time() - start_time

            if j % train_params['eval_freq'] == 0:
                evaluate_model(i, j, model, the_categories, corpus, train_params, training_took, loss_sum, tokens_sum)
                loss_sum = 0
                tokens_sum = 0

    return performance_dict


if __name__ == "__main__":
    main()
