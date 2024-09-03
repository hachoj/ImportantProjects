import matplotlib.pyplot as plt
from collections import OrderedDict

test = OrderedDict({"w": [0, 1, 2, 3], "b": [0, 0.5, 1, 1.5]})
plt.plot(test["w"])
plt.plot(test["b"])
plt.savefig("ViT/plots/test.png")