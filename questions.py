from ex2 import *
import matplotlib.pyplot as plt

def question1(portions):
    print("\nSingle layer MLP results:")

    epoch_losses = []

    def callback(loss):
        nonlocal epoch_losses
        epoch_losses.append(loss)


    for p in portions:
        perceptron = Perceptron(input_dim=FEATURE_DIM)
        accuracy = MLP_classification(portion=p, model=perceptron, callback=callback)
        # Plot
        plt.plot(epoch_losses)


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    # Q1 - single layer MLP
    question1(FEATURE_DIM, MLP_classification, portions)


    # Q2 - multi-layer MLP
    pass

    # Q3 - Transformer
    print("\nTransformer results:")
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_classification(portion=p)

