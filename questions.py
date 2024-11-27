from ex2 import *
import matplotlib.pyplot as plt

def question1(portions):
    print("\nSingle layer MLP results:")

    for p in portions:
        perceptron = Perceptron()
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
    question1(portions)


    # Q2 - multi-layer MLP
    pass

    # Q3 - Transformer
    # print("\nTransformer results:")
    # for p in portions[:2]:
    #     print(f"Portion: {p}")
    #     transformer_classification(portion=p)

