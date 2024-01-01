try:
    from progdiff.distributional_models.datasets.childes import Childes
except:
    from distributional_models.datasets.childes import Childes


def create_corpus(corpus_path, language="eng", collection_name=None, age_range=(0, 1000), sex_list=None,
                  add_punctuation=True, exclude_target_child=True, num_documents=0):

    the_corpus = Childes()
    the_corpus.get_documents_from_childes_db_file(input_path=corpus_path,
                                                  language=language,
                                                  collection_name=collection_name,
                                                  age_range_tuple=age_range,
                                                  sex_list=sex_list,
                                                  add_punctuation=add_punctuation,
                                                  exclude_target_child=exclude_target_child,
                                                  num_documents=num_documents)

    return the_corpus


def main():

    input_path = "../input_data/childes.csv"
    the_corpus = create_corpus(input_path)
    the_corpus.save_to_pkl_file("input_data/childes")

    return the_corpus


if __name__ == "__main__":
    main()
