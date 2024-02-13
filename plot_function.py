from matplotlib import pyplot as plt


def plot_the_model(predictions, labels):
    """Plot the predicted charges against the actual price."""
    plt.scatter(labels, predictions)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Actual vs. Predicted Ratings")
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""
    plt.plot(epochs, rmse, label="Root Mean Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Root Mean Squared Error vs. Epoch")
    plt.legend()
    plt.show()

