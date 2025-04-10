import torch
import torch.nn as nn


# ✅ Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# ✅ ACNN Block
class ACNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ACNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_attention(x)  # Apply channel attention
        return x

# ✅ ACNN-BiLSTM Model (Handles Subjects with 56 Windows)
class ACNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=12, cnn_filters=64, lstm_hidden_size=128, num_lstm_layers=2):
        super(ACNN_BiLSTM, self).__init__()

        # Initial Conv2D before ACNN blocks
        self.conv_init = nn.Conv2d(input_channels, cnn_filters, kernel_size=3, stride=1, padding=1)

        # ACNN Blocks (7x)
        self.acnn_blocks = nn.Sequential(
            ACNNBlock(cnn_filters, cnn_filters),
            ACNNBlock(cnn_filters, cnn_filters),
            ACNNBlock(cnn_filters, cnn_filters),
            ACNNBlock(cnn_filters, cnn_filters),
            ACNNBlock(cnn_filters, cnn_filters),
            ACNNBlock(cnn_filters, cnn_filters),
            ACNNBlock(cnn_filters, cnn_filters),
        )

        # BiLSTM for Temporal Feature Learning
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers,
                            batch_first=True, bidirectional=True)

        # Fully Connected Layers (FCN)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 128)  # BiLSTM outputs (2 * hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)  # Output SBP & DBP

    def forward(self, x):
        """
        Input: (batch=3, 56, 127, 1000, 12)
        Output: (batch=3, 2) [SBP, DBP]
        """
        batch_size, num_windows, C, H, W = x.shape
        x = x.view(batch_size * num_windows, C, H, W)  # Merge batch & window_num for CNN

        # Initial Conv2D
        x = self.conv_init(x)

        # 7 ACNN Blocks
        x = self.acnn_blocks(x)

        # Global Average Pooling across spatial dimensions
        x = torch.mean(x, dim=(2, 3))  # (batch * num_windows, C)

        # Reshape for LSTM: (Batch=3, Windows=56, Features=CNN_Output_Size)
        x = x.view(batch_size, num_windows, -1)

        # BiLSTM
        x, _ = self.lstm(x)  # (batch, num_windows, 2 * hidden_size)

        # Extract last LSTM output per subject
        x = x[:, -1, :]  # Take last time step for each subject

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)  # Output 2 values (SBP & DBP)

        return x  # Shape: (batch=3, 2)

if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = ACNN_BiLSTM().to(device)

    # Create a dummy test input (Batch=3 subjects, each with 56 time windows)
    dummy_input = torch.randn(3, 56, 127, 1000, 12).to(device)

    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)  # Forward pass

    # Print output shape
    print("Model Output Shape:", output.shape)  # Expected: (3, 2) [SBP, DBP]

# Define a dataset class
