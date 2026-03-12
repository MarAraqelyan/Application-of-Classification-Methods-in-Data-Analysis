import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df.hist(figsize=(10, 8))
plt.suptitle("Iris Dataset Histogram")
plt.savefig("Iris_histogram.png")
plt.show()