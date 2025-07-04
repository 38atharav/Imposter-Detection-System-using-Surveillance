import torch.nn as nn
from torchvision.models import mobilenet_v2

class LRCN(nn.Module):
    def __init__(self, num_classes, lstm_hidden=128, lstm_layers=1):
        super(LRCN, self).__init__()
        # CNN Feature Extractor with adjusted output
        base = mobilenet_v2(pretrained=True)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        
        # Adaptive pooling to fix output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LSTM (input size matches CNN output)
        self.lstm = nn.LSTM(1280, lstm_hidden, lstm_layers, batch_first=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, C, H, W)
        batch, seq_len = x.shape[:2]
        
        # CNN features
        c_in = x.view(batch * seq_len, *x.shape[2:])
        c_out = self.cnn(c_in)
        c_out = self.adaptive_pool(c_out)
        c_out = c_out.view(batch, seq_len, -1)  # Now shape (batch, seq_len, 1280)
        
        # LSTM
        lstm_out, _ = self.lstm(c_out)
        
        # Classifier
        return self.fc(lstm_out[:, -1, :])