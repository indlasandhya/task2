import matplotlib.pyplot as plt

def plot_results(y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Plot")
    plt.show()
