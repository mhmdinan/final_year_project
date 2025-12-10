import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

# Load data (for example, Iris dataset)
data = load_iris()
X = data['data']
X = np.array(X)

# Standardizing the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_normalized_tensor = torch.FloatTensor(X_normalized)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Hyperparameters
input_dim = X_normalized.shape[1]  # Original feature dimension (4 for Iris)
encoding_dim = 2  # You can adjust the size of the encoding
num_epochs = 1000
learning_rate = 0.05

# Create the autoencoder model
model = Autoencoder(input_dim, encoding_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the autoencoder
for epoch in range(num_epochs):
    model.train()
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_normalized_tensor)
    
    # Compute the loss
    loss = criterion(outputs, X_normalized_tensor)
    
    # Backward pass
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss at every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Reconstruct the original data
model.eval()
with torch.no_grad():
    X_reconstructed_tensor = model(X_normalized_tensor)

X_reconstructed = X_reconstructed_tensor.numpy()

# Calculate MSE
mse = mean_squared_error(X_normalized, X_reconstructed)
print(f'Mean Squared Error: {mse}')

# Calculate R²
variance_original = np.var(X_normalized)
r_squared = 1 - (mse / variance_original)
print(f'R-squared: {r_squared}')