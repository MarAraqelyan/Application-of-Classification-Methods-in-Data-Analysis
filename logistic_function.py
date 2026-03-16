import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 400)
p = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, p)
plt.xlabel("z")
plt.ylabel("P(Y=1|X)")
plt.title("Logistic Function")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("logistic_function.png", dpi=300, bbox_inches="tight")
plt.savefig("images/logistic_function.png")
plt.show()