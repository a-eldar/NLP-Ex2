from ex2 import *
import matplotlib.pyplot as plt
# from torch.nn import Perceptron

def question1(portions):
    print("\nSingle layer MLP results:")

    for p in portions:
        perceptron = MLPClassifier(
            hidden_layer_sizes=[], # single layer
            solver='adam',
            batch_size=16,
            learning_rate_init=0.001,
        )
        _, epoch_losses, accuracies = MLP_classification(portion=p, model=perceptron)
        plt.plot(epoch_losses, color='red', label='Loss')
        plt.plot(accuracies, color='blue', label='Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss and accuracy')
        plt.title('Loss vs Epoch')
        plt.show()

def question2(portions):
    print("\nMulti-layer MLP results:")
    for p in portions:
        perceptron = MLPClassifier(
            hidden_layer_sizes=[500], # single hidden layer
            solver='adam',
            batch_size=16,
            learning_rate_init=0.001,
        )
        _, epoch_losses, accuracies = MLP_classification(portion=p, model=perceptron)
        plt.plot(epoch_losses, color='red', label='Loss')
        plt.plot(accuracies, color='blue', label='Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss and accuracy')
        plt.title('Loss vs Epoch')
        plt.show()

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

