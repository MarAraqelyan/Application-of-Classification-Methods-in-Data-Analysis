import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.grid(True)
plt.title("Iris Dataset boxplot")
plt.savefig("Iris_boxplot.png")
plt.show()