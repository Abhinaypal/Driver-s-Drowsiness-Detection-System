"""
LSTM model for temporal drowsiness detection from feature sequences.
"""
from typing import Tuple

import torch
from torch import nn


class LSTMDrowsinessDetector(nn.Module):
    """
    LSTM-based temporal drowsiness detector.
    
    Processes sequences of feature vectors (PERCLOS, head pose, eye state)
    over time to detect temporal patterns of drowsiness.
    
    Architecture:
    - Input: Sequence of feature vectors (batch_size, seq_len, input_size)
    - LSTM layers: Capture temporal dependencies
    - Output: Classification of drowsiness state
    
    Classes:
    - 0: Awake
    - 1: Drowsy/Microsleep
    - 2: Asleep
    """
    
    def __init__(
        self,
        input_size: int = 10,      # Number of features per frame
        hidden_size: int = 64,      # LSTM hidden size
        num_layers: int = 2,        # Number of LSTM layers
        num_classes: int = 3,       # Output classes
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classification head
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len, input_size)
                      containing feature vectors over time
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(sequences)
        
        # Use last hidden state for classification
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        
        # Classification
        logits = self.classifier(last_hidden)
        return logits
    
    def predict_proba(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(sequences)
            return torch.softmax(logits, dim=1)


class GRUDrowsinessDetector(nn.Module):
    """
    GRU-based temporal drowsiness detector (lighter-weight LSTM alternative).
    
    GRU has fewer parameters than LSTM, making it faster while still
    capturing temporal dependencies effectively.
    
    Classes:
    - 0: Awake
    - 1: Drowsy/Microsleep
    - 2: Asleep
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classification head
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        gru_out, h_n = self.gru(sequences)
        
        # Use last hidden state
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        
        logits = self.classifier(last_hidden)
        return logits
    
    def predict_proba(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(sequences)
            return torch.softmax(logits, dim=1)
