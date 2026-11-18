"""
CNN-BiLSTM-Attention model for SOC estimation.

Architecture follows the structure reported in recent low-temperature-focused
SOC estimation research (e.g. "State-of-Charge Estimation of Lithium-Ion
Batteries Based on the CNN-Bi-LSTM-AM Model Under Low-Temperature
Environments", PMC, 2025):

    1D-CNN front end  -> extracts local patterns from the voltage/current/
                         temperature/power sequence within each window.
    Bidirectional LSTM -> captures temporal dependencies in both directions
                          within the window (not just past -> present, unlike
                          the plain unidirectional LSTM baseline).
    Attention layer    -> learns which timesteps in the window matter most
                          for the final SOC estimate, instead of only using
                          the last timestep's hidden state.
    Linear head        -> single SOC output.

Same input/output contract as SoCLSTM: input [batch, seq_len, n_features],
output [batch] (or [batch, 1], squeezed consistently with build_model usage
in train.py). input_size defaults to 4, matching FEATURE_COLUMNS in
src/data/dataset.py (Voltage, Current, Temperature, Power).
"""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Simple additive (Bahdanau-style) attention over a sequence of hidden
    states. Learns a per-timestep importance score, normalized with softmax,
    then returns the weighted sum (context vector) -- so the model can focus
    on the timesteps that matter most for this window's SOC, rather than
    being forced to rely only on the final timestep."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        # seq: [batch, seq_len, hidden_size]
        scores = self.score(seq).squeeze(-1)          # [batch, seq_len]
        weights = torch.softmax(scores, dim=1)          # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), seq)  # [batch, 1, hidden_size]
        return context.squeeze(1)                        # [batch, hidden_size]


class CNNBiLSTMAttention(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        cnn_channels: int = 32,
        cnn_kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 'same' padding so the CNN preserves the sequence length -- the
        # BiLSTM still needs one hidden state per original timestep.
        padding = cnn_kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=cnn_kernel_size, padding=padding),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2  # *2 for bidirectional
        self.attention = AdditiveAttention(lstm_out_size)
        self.head = nn.Linear(lstm_out_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = x.transpose(1, 2)        # -> [batch, input_size, seq_len] for Conv1d
        x = self.cnn(x)              # -> [batch, cnn_channels, seq_len]
        x = x.transpose(1, 2)        # -> [batch, seq_len, cnn_channels] for LSTM

        lstm_out, _ = self.lstm(x)   # -> [batch, seq_len, hidden_size * 2]
        context = self.attention(lstm_out)  # -> [batch, hidden_size * 2]

        soc = self.head(context).squeeze(-1)  # -> [batch]
        return soc