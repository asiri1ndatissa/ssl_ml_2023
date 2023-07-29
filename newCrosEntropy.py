import torch
import torch.nn.functional as TF

# Define some sample input data and labels
input_data = torch.randn(4, 10)  # 4 samples, 10 classes
labels = torch.LongTensor([2, 5, 1, 9])  # target class indices

# Compute the cross entropy loss
loss = TF.cross_entropy(input_data, labels)

# Print the computed loss
print(f"Cross entropy loss: {loss.item()}")

# Compute the softmax probabilities manually
softmax_probs = TF.softmax(input_data, dim=1)

# Print the computed softmax probabilities
print(f"Softmax probabilities:\n{softmax_probs}")

# Compute the cross entropy loss manually
manual_loss = torch.mean(-torch.log(softmax_probs.gather(1, labels.view(-1,1)) ))

# Print the manually computed loss
print(f"Manually computed loss: {manual_loss.item()}")

predicted_probs = softmax_probs[torch.arange(len(labels)), labels]

print(predicted_probs)
