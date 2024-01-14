from distributional_models.corpora.childes import Childes
from distributional_models.tasks.categories import Categories
from distributional_models.scripts.create_model import create_model
from distributional_models.scripts.evaluate_model import evaluate_model


def main():
    import params
    run_progdiff(params.param2default, "local")


def run_progdiff(param2val, run_location):
    param2val = fix_paths(param2val, run_location)

    the_categories = Categories()
    the_categories.create_from_category_file(param2val['category_file_path'])

    the_corpus, missing_words = init_corpus(param2val,
                                            the_categories.instance_list)
    the_categories.remove_instances(missing_words)

    the_model = create_model(the_corpus.vocab_size, param2val)

    performance_dict = train_model(the_corpus, the_model, the_categories, param2val)
    return performance_dict


def fix_paths(train_params, run_location):
    if run_location == 'local':
        train_params['corpus_path'] = "../" + train_params['corpus_path']  # "input_data/childes"
        train_params['category_file_path'] = "../" + train_params['category_file_path']  # "input_data/categories"
    elif run_location == 'ludwig_cluster':
        train_params['corpus_path'] = "/media/ludwig_data/Progdiff/" + train_params['corpus_path']
        train_params['category_file_path'] = "/media/ludwig_data/Progdiff/" + train_params['category_file_path']
        # TODO fix save_path for ludwig so it ends up in the same runs folder
    else:
        raise ValueError(f"Unrecognized run location {run_location}")

    return train_params


def init_corpus(params, include_list):
    # TODO this can be improved to catch specific exceptions, like is the file there, and the error that will occur
    # if you loaded a childes instance created under a different path
    try:
        the_corpus = Childes.load_from_file(params['corpus_path']+'.pkl')
    except:
        the_corpus = Childes()
        the_corpus.get_documents_from_childes_db_file(input_path=params['corpus_path'],
                                                      language='eng',
                                                      collection_name=None,
                                                      age_range_tuple=(0,1000),
                                                      sex_list=None,
                                                      add_punctuation=True,
                                                      exclude_target_child=True,
                                                      num_documents=0)
        the_corpus.save_to_pkl_file(params['corpus_path'])
    missing_words = the_corpus.create_vocab(vocab_size=params['vocab_size'],
                                            include_list=include_list,
                                            include_unknown=True)
    the_corpus.batch_docs_by_age(params['num_document_batches'], params['document_order'])
    return the_corpus, missing_words


def train_model(corpus, model, the_categories, train_params):
    performance_dict = {}
    took_sum = 0

    for j in range(len(corpus.document_list)):
        print(f"Training corpus chunk {j+1}/{len(corpus.document_list)} for {train_params['num_epochs']} epochs")
        for i in range(train_params['num_epochs']):
            sequence = corpus.document_list[j]

            loss_mean, took = model.train_sequence(corpus, sequence, train_params)
            took_sum += took

            if j % train_params['eval_freq'] == 0:
                took_mean = took_sum / train_params['eval_freq']
                took_sum = 0
                output_string = evaluate_model(f"d{j}_e{i}", model, the_categories, corpus, train_params, took_mean,
                                               loss_mean)
                print(output_string)

            if j % train_params['save_freq'] == 0:
                file_name = f"d{j}_e{i}.pth"
                model.save_model(train_params['save_path'], file_name)

    return performance_dict


if __name__ == "__main__":
    main()
