import torch
import matplotlib.pyplot as plt

model = torch.load('outputs/basic/model_000500.pth')
print(model.total_loss)



plt.ylim([0.9, 1.0])
utils.plot_loss(train_accuracy, "Loss")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.legend()
plt.xlabel("Number of gradient steps")
plt.ylabel("Total loss")
plt.savefig("softmax_train_graph.png")
plt.show()