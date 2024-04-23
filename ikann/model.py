import torch
import torch.nn as nn
import torch.optim as optim

class IKAnnModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(IKAnnModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example usage:
input_size = 3  # Input size is the dimension of the desired end effector velocity (3 DoF)
output_size = 3 * 6  # Output size is the dimension of the joint positions (6 joints * 3 DoF)
hidden_size = 128  # Hidden layer size

# Create an instance of the neural network
# model = NeuralNetwork(input_size, output_size, hidden_size)

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Example input (desired end effector velocity)
# input_velocity = torch.randn(1, input_size)

# # Forward pass
# output_joint_positions = model(input_velocity)

# print("Output joint positions:", output_joint_positions)

