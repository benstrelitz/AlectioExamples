import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        # print("\n")

        # print("Input size:",text.size())

        embedded = self.dropout(self.embedding(text))

        # print("Embedded size:", embedded.size())

        output, (hidden, cell) = self.rnn(embedded)

        # print("Hidden output size:", hidden.size())

        # print("Output size:", output.size())

        # pull out the last 2 layers from the RNN hidden layers and concatonate them to form our recurrent feature set.
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # print("Hidden output size (post concat)", hidden.size())

        output = self.fc(hidden)

        # print("output size", output.size())

        # print("\n")

        return output
