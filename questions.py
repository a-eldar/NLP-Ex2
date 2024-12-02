from ex2 import *
import matplotlib.pyplot as plt
# from torch.nn import Perceptron

LOSS_COLOR = 'red'
ACC_COLOR = 'blue'

def plot_loss_and_acc(epoch_losses, epoch_accuracies, portion):
    plt.plot(epoch_losses, color=LOSS_COLOR)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch, Portion: {portion}')
    plt.show()

    plt.plot(epoch_accuracies, color=ACC_COLOR)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Epoch, Portion: {portion}')
    plt.show()


def question1(portions):
    print("\nSingle layer MLP results:")

    for p in portions:
        perceptron = MLPClassifier(
            hidden_layer_sizes=[], # single layer
            solver='adam',
            batch_size=16,
            learning_rate_init=0.001,
        )
        _, epoch_losses, epoch_accuracies = MLP_classification(portion=p, model=perceptron)
        plot_loss_and_acc(epoch_losses, epoch_accuracies, p)


def question2(portions):
    print("\nMulti-layer MLP results:")
    for p in portions:
        perceptron = MLPClassifier(
            hidden_layer_sizes=[500], # single hidden layer
            solver='adam',
            batch_size=16,
            learning_rate_init=0.001,
        )
        _, epoch_losses, epoch_accuracies = MLP_classification(portion=p, model=perceptron)
        plot_loss_and_acc(epoch_losses, epoch_accuracies, p)

if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    # Q1 - single layer MLP
    # question1(portions)


    # Q2 - multi-layer MLP
    question2(portions)

    # Q3 - Transformer
    # print("\nTransformer results:")
    # for p in portions[:2]:
    #     print(f"Portion: {p}")
    #     transformer_classification(portion=p)

