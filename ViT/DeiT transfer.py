from transformers import DeiTForImageClassification, DeiTImageProcessor
import torch
import torch.nn as nn
from datasets import load_dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

dataset = load_dataset("CIFAR10")
model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float32)

model.classifier = nn.Linear(model.config.hidden_size, 10)
# for some reason initializig the classifier's weights and biases with kaiming and zeros made the initial
# loss worse rather than better so I'm just going to let PyTorch do its thing
# kaiming initialization for the classifier
# nn.init.kaiming_normal_(model.classifier.weight)
# nn.init.zeros_(model.classifier.bias)
model.num_labels = 10

model.to(device)

id_to_lb = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

image_test = dataset["test"][0]
plt.imshow(image_test["img"])
plt.title(id_to_lb[image_test["label"]])
plt.show()

image_processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")


class Dataloader:
    def __init__(self, dataset, image_processor, B):
        self.B = B
        self.dataset = dataset
        self.image_processor = image_processor
        self.current_idx = 0

    def shuffle(self):
        self.dataset = self.dataset.shuffle()

    def get_batch(self):
        images = self.dataset["img"][self.current_idx:self.current_idx+self.B]
        labels = self.dataset["label"][self.current_idx:self.current_idx+self.B]
        images = self.image_processor(images, return_tensors="pt")
        labels = torch.tensor(labels).to(device)
        images = {k: v.to(device) for k, v in images.items()}
        self.current_idx += self.B
        if self.current_idx >= len(self.dataset["img"]):
            self.current_idx = 0
            self.shuffle()
        return images, labels

dl = Dataloader(dataset["train"], image_processor, B=1024)
vl = Dataloader(dataset["test"], image_processor, B=128)

# before training, we freeze the model weights, except for the classifier

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# cosine learning rate schedule
def lr_sched(iter, min_lr, max_lr, T):
    if iter < (5*T)/6:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(iter) * 3.14159 / T))
    else:
        return 1e-4
    


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

iters = 20
lossi = []
acc = []
grad_norms = {"w": [], "b": []}

for iter in range(iters):
    optimizer.param_groups[0]["lr"] = lr_sched(iter, 1e-6, 1.5e-2, iters)
    images, labels = dl.get_batch()
    outputs = model(images["pixel_values"], labels=labels)
    loss = outputs.loss
    lossi.append(loss.item())
    loss.backward()
    # L2 regularization
    # for param in model.classifier.parameters():
    #     param.grad += 1e-4 * param
    # print the grad norm of the classifier
    w_n = torch.norm(model.classifier.weight.grad)
    b_n = torch.norm(model.classifier.bias.grad)
    grad_norms["w"].append(w_n.item())
    grad_norms["b"].append(b_n.item())
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        images, labels = vl.get_batch()
        outputs = model(images["pixel_values"], labels=labels)
    print(f"iter: {iter}, loss: {loss.item()}, accuracy: {torch.mean((torch.argmax(outputs.logits, dim=1) == labels).float()).item()}, w_norm: {w_n.item()}, b_norm: {b_n.item()}")
    acc.append(torch.mean((torch.argmax(outputs.logits, dim=1) == labels).float()).item())
    if iter == 5:
        break

# print the model predictions with matplotlib
# plot the loss and accuracy
plt.clf()
plt.plot(lossi)
plt.title("Loss")
plt.savefig("ViT/plots/loss.png")
plt.clf()
plt.plot(acc)
plt.title("Accuracy")
plt.savefig("ViT/plots/accuracy.png")
plt.clf()
plt.plot(grad_norms["w"])
plt.plot(grad_norms["b"])
plt.savefig("ViT/plots/grad_norms.png")

# images, labels = dl.get_batch()
# outputs = model(images["pixel_values"])
# preds = torch.argmax(outputs.logits, dim=1)
# for i in range(len(preds)):
#     plt.imshow(images["pixel_values"][i].cpu().numpy().transpose(1, 2, 0))
#     plt.title(f"pred: {id_to_lb[preds[i].item()]}, true: {id_to_lb[labels[i].item()]}")
#     plt.show()

torch.save(model.state_dict(), "ViT/DeiT_Transfer_CIFAR-10.pth")