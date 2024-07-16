
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def custom_collate_fn(batch):
    batch.sort(key=lambda x: x['mel_length'], reverse=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ids = [item['id'] for item in batch]
    phonemes = [item['phonemes_encoded'] for item in batch]
    mel_features = [item['mel_features'] for item in batch]
    lengths = [item['mel_length'] for item in batch]
    stop_tokens=[item['stop_tokens'] for item in batch]

    phonemes_padded = torch.nn.utils.rnn.pad_sequence(phonemes, batch_first=True, padding_value=0).to(device)


    max_len = max(lengths)
    mel_specs_padded = torch.stack([torch.cat([mel, torch.zeros(size=(mel.size(0), max_len - mel.size(1)))], dim=1) for mel in mel_features])

    stop_tokens_padded = torch.stack([torch.cat([stop, torch.zeros(max_len - stop.size(0))], dim=0) for stop in stop_tokens])

    # Convert lengths to tensor
    phoneme_seq_padded = phonemes_padded.to(device, non_blocking=True).long()
    mel_specs_padded = mel_specs_padded.to(device, non_blocking=True).long()
    stop_tokens_padded = stop_tokens_padded.to(device, non_blocking=True).long()

    batch_padded = {
        'id': ids,
        'phonemes_encoded': phoneme_seq_padded,
        'mel_features': mel_specs_padded.transpose(1, 2),
        'length': torch.tensor(lengths, device=device).long(),
        'stop_tokens': stop_tokens_padded
    }

    return batch_padded




class EncoderPrenet(nn.Module):
    def __init__(self, embedding_dim=512, num_channels=512, kernel_size=3, dropout_rate=0.5):
        super(EncoderPrenet, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, num_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size, padding=1)
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(num_channels)
        self.batch_norm2 = nn.BatchNorm1d(num_channels)
        self.batch_norm3 = nn.BatchNorm1d(num_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_channels, embedding_dim)

    def forward(self, x):
        # Assume x is of shape (batch_size, seq_length, embedding_dim)
        x = x.transpose(1, 2)
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.dropout(x)
        # Transpose back to (batch_size, seq_length, num_channels)
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x



class DecoderPrenet(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=256, final_dim=512, dropout_rate=0.5):
        super(DecoderPrenet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.linear_projection = nn.Linear(output_dim, final_dim) # Added linear projection
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.linear_projection(x)  # Apply the linear projection
        return x


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ScaledPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(1))

        # Create constant positional encoding matrix with max_len positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add positional encoding to each sequence element, scaled by the trainable weight
        x = x + self.scale * self.pe[:, :x.size(1)]
        return x


