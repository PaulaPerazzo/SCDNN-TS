# Importando as bibliotecas do PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaselineCNN1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5):
        super(BaselineCNN1D, self).__init__()
        
        # O in_channels padrão é 12 (as 12 derivações do ECG do PTB-XL)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        
        # Adaptive Pooling: Reduz qualquer comprimento de sequência para 1 valor por filtro
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x