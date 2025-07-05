# class CustomCNNBiLSTMAttentionFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: Space, features_dim: int = 256):
#         super(CustomCNNBiLSTMAttentionFeatureExtractor, self).__init__(
#             observation_space, features_dim
#         )
#         n_time_steps, n_features = observation_space.shape
#
#         # Parallel Conv1d layers (inception-style)
#         self.cnn_2 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=2, padding="same")
#         self.cnn_4 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=4, padding="same")
#         self.cnn_8 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=8, padding="same")
#         self.bn = nn.BatchNorm1d(192)
#         self.relu = nn.ReLU()
#         self.cnn_dropout = nn.Dropout(p=0.2)
#
#         # LSTM layer
#         self.lstm = nn.LSTM(
#             input_size=192,
#             hidden_size=128,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.2,
#         )
#
#         # Multihead Attention layer
#         self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
#
#         # Final dropout before flattening output
#         self.final_dropout = nn.Dropout(p=0.2)
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         # Input shape: (batch, time, features)
#         x = observations  # shape (batch, time, features)
#         x = x.permute(0, 2, 1)  # (batch, features, time)
#
#         # Inception-style parallel convs
#         x2 = self.cnn_2(x)
#         x4 = self.cnn_4(x)
#         x8 = self.cnn_8(x)
#         x = th.cat([x2, x4, x8], dim=1)  # (batch, 192, time)
#
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.cnn_dropout(x)
#
#         x = x.permute(0, 2, 1)  # (batch, time, channels=192)
#         lstm_out, _ = self.lstm(x)  # (batch, time, 256)
#
#         # MultiheadAttention expects (batch, time, features)
#         attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, time, 256)
#
#         # Weighted sum over time steps
#         attn_weights = F.softmax(attn_out.mean(dim=2), dim=1)  # (batch, time)
#         attended = (attn_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, 256)
#
#         attended = self.final_dropout(attended)
#         return attended
