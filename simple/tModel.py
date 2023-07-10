import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout),
            num_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout),
            num_decoder_layers
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, keypoints):
        positional_encoded = self.positional_encoding(keypoints)

        encoder_output = self.encoder(positional_encoded)
        decoder_output = self.decoder(positional_encoded, encoder_output)

        output = self.fc(decoder_output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=100):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
