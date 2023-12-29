try:
    from distributional_models.models.neural_network import NeuralNetwork
    from distributional_models.tasks.cohyponym_task import CohyponymTask
    from distributional_models.datasets.childes import Childes
except:
    from progdiff.distributional_models.models.neural_network import NeuralNetwork
    from progdiff.distributional_models.tasks.cohyponym_task import CohyponymTask
    from progdiff.distributional_models.datasets.childes import Childes


def create_corpus(corpus_path, vocab_size, target_list):

    language = "eng"
    collection_name = None
    age_range_tuple = (0, 1000)
    sex_list = None
    add_punctuation = True
    exclude_target_child = True
    num_documents = 0

    the_corpus = Childes()
    the_corpus.get_documents_from_childes_db_file(input_path=corpus_path,
                                                  language=language,
                                                  collection_name=collection_name,
                                                  age_range_tuple=age_range_tuple,
                                                  sex_list=sex_list,
                                                  add_punctuation=add_punctuation,
                                                  exclude_target_child=exclude_target_child,
                                                  num_documents=num_documents)

    the_corpus.create_vocab(vocab_size=vocab_size, include_list=target_list, include_unknown=True)
    return the_corpus


def main():

    #  load the category file
    category_file_path = "dataset/categories.csv"
    evaluation_layer = 0
    sequence_list = None
    target_list, target_index_dict = CohyponymTask.load_category_file(category_file_path)

    # load the corpus and create the vocab
    vocab_size = 1024
    childes_db_path = 'dataset/raw_childes.csv'
    childes_pkl_path = 'dataset/childes.pkl'
    # corpus = create_corpus(childes_db_path, vocab_size, target_list)
    corpus = Childes.load_from_file(childes_pkl_path)
    corpus.create_vocab(vocab_size=vocab_size, include_list=target_list, include_unknown=True)

    # set model architecture parameters
    embedding_size = 0
    hidden_layer_info_list = [("lstm", 256)]
    weight_init = 0.0001
    window_size = None
    device = 'cpu'
    criterion = 'cross_entropy'  # loss function
    model = NeuralNetwork(embedding_size, hidden_layer_info_list, weight_init, corpus.vocab_index_dict, criterion,
                          device=device)
    print(model.layer_list)

    num_epochs = 1
    learning_rate = 0.1

    optimizer = 'adagrad'

    cohyponym_task = CohyponymTask(model,
                                   evaluation_layer,
                                   category_file_path,
                                   sequence_list=sequence_list,
                                   num_thresholds=21)

    for i in range(num_epochs):
        for i in range(corpus.num_documents):
            doc_sequence_list = corpus.flatten_corpus_lists(corpus.document_list[i])
            corpus.x_list, corpus.y_list = corpus.create_sequence_list(doc_sequence_list,
                                                                       corpus.vocab_index_dict,
                                                                       corpus.unknown_token,
                                                                       window_size=window_size)
            took, perplexity = model.train_sequence(corpus.x_list, corpus.y_list, criterion, optimizer, learning_rate)
            mean_ba = cohyponym_task.run_cohyponym_task()
            print(f"doc: {i}   took: {took:0.2f}   perp: {perplexity:0.2f}   ba: {mean_ba:0.2f}")


if __name__ == "__main__":
    main()
