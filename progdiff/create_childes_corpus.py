from distributional_models.datasets.childes import Childes


def main():

    input_path = "dataset/raw_childes.csv"

    language = "eng"
    collection_name = None
    age_range_tuple = (0, 1000)
    sex_list = None
    add_punctuation = True
    exclude_target_child = True

    the_corpus = Childes()

    the_corpus.get_documents_from_childes_db_file(input_path=input_path,
                                                  language=language,
                                                  collection_name=collection_name,
                                                  age_range_tuple=age_range_tuple,
                                                  sex_list=sex_list,
                                                  add_punctuation=add_punctuation,
                                                  exclude_target_child=exclude_target_child)

    the_corpus.save_to_file("dataset/childes.pkl")


main()
