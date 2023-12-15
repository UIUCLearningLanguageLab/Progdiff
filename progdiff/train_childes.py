from distributional_models.models.neural_network import NeuralNetwork
from distributional_models.datasets.childes import Childes
import torch
from distributional_models.tasks.cohyponym_task import CohyponymTask


def main():
    vocab_size = 4096

    embedding_size = 64
    hidden_layer_type_list = ["lstm"]
    hidden_layer_size_list = [512]

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    learning_rate = 0.1
    weight_init = 0.0001

    num_epochs = 1

    evaluation_layer = 0
    token_category_dict = {'.': 'PUNCT',
                           '?': 'PUNCT',
                           '!': 'PUNCT',
                           'you': 'PRON',
                           'i': 'PRON',
                           'he': 'PRON',
                           'she': 'PRON',
                           'mommy': 'NAME',
                           'daddy': 'NAME',
                           'baby': 'NAME',
                           'dog': 'ANIMAL',
                           'cat': 'ANIMAL',
                           'mouse': 'ANIMAL',
                           'run': 'ACTION_VERB',
                           'jump': 'ACTION_VERB',
                           'eat': 'ACTION_VERB',
                           'drink': 'ACTION_VERB',
                           }

    childes_corpus = Childes.load_from_file("dataset/childes.pkl")
    childes_corpus.create_vocab(vocab_size=vocab_size, include_unknown=True)

    model = NeuralNetwork(embedding_size, hidden_layer_type_list, hidden_layer_size_list, weight_init, device,
                          childes_corpus.vocab_index_dict)
    print(model.layer_list)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.to(device)

    for i in range(num_epochs):
        for j in range(childes_corpus.num_documents):
            doc_sequence_list = childes_corpus.flatten_corpus_lists(childes_corpus.document_list[j])

            childes_corpus.sequence_list = childes_corpus.create_sequence_list(doc_sequence_list,
                                                                               childes_corpus.vocab_index_dict,
                                                                               childes_corpus.unknown_token)
            took, perplexity = model.train_sequence(childes_corpus.sequence_list, criterion, optimizer)

            cohyponym_task = CohyponymTask(model, evaluation_layer, token_category_dict, sequence_list=None,
                                           num_thresholds=21)
            ba = cohyponym_task.best_category_ba_df['ba_mean'].mean()
            cohyponym_task.save_results("results/childes_ba.csv")

            print(f"doc: {j}/{childes_corpus.num_documents}   took: {took:0.2f}   perplexity: {perplexity:0.2f}   BA: {ba}")


if __name__ == "__main__":
    main()
