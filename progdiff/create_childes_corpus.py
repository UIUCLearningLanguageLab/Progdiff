try:
    from progdiff.distributional_models.datasets.childes import Childes
except:
    from distributional_models.datasets.childes import Childes


def main():

    input_path = "dataset/raw_childes.csv"

    language = "eng"
    collection_name = None
    age_range_tuple = (0, 1000)
    sex_list = None
    add_punctuation = True
    exclude_target_child = True
    num_documents = 10

    the_corpus = Childes()

    the_corpus.get_documents_from_childes_db_file(input_path=input_path,
                                                  language=language,
                                                  collection_name=collection_name,
                                                  age_range_tuple=age_range_tuple,
                                                  sex_list=sex_list,
                                                  add_punctuation=add_punctuation,
                                                  exclude_target_child=exclude_target_child,
                                                  num_documents=num_documents)
    print(the_corpus)
    print(the_corpus.document_info_df)
    the_corpus.save_to_csv_file("dataset/childes")
    #the_corpus.save_to_csv_file("dataset/childes.csv")

    return the_corpus


main()
