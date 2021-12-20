import torch
import torch.nn
from dataclasses import dataclass

@dataclass
class RNNOutput(object):
    """
    This class is used to store the output of a RNN.
    """
    loss: torch.Tensor
    logits: torch.Tensor
    aux_loss: float = 0.

class DecoderModel(torch.nn.Module):
    def __init__(self, input_seq_len, target_seq_len, hidden_size, tokenizer):
        super().__init__()
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.first_decoder = torch.nn.Linear(input_seq_len, target_seq_len)
        self.second_decoder = torch.nn.Linear(self.hidden_size, self.tokenizer.vocab_size)
        self.act = torch.nn.ReLU()
        self.init_weights()

    def forward(self, ctx_vec, **kwargs):
        # out: (batch_size, seq_len, hidden_size)
        # first_decoder: (target, input) (transpose)
        # decoder_out: (batch_size, target, hidden_size)
        out1 = torch.einsum("bih, ti -> bth", ctx_vec, self.first_decoder.weight)
        out2 = torch.einsum("bth, vh -> btv", out1, self.second_decoder.weight)
        # out3 = self.act(out2)
        return out2

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.first_decoder.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.second_decoder.weight, gain=1.0)
        self.first_decoder.bias.data.zero_()
        self.second_decoder.bias.data.zero_()

class EncoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class RNNModel(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_seq_len, target_seq_len_, tokenizer):
        super(RNNModel, self).__init__()
        self.tokenizer = tokenizer
        self.input_size = 32
        self.hidden_size = 64
        self.output_size = 32
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self.input_size)
        self.input_seq_len = input_seq_len
        self.target_seq_len_ = target_seq_len_
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder = DecoderModel(input_seq_len, target_seq_len_, self.hidden_size, tokenizer)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def loss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        # print(f"logits.min: {logits.min()}, logits.max(): {logits.max()}")
        loss = loss_fct(logits, labels)
        return loss

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.embedding.weight, gain=1.0)

    def init_hidden_states(self, batch_size):
        hidden_state = torch.zeros(1, batch_size, self.hidden_size)
        cell_state = torch.zeros(1, batch_size, self.hidden_size)
        torch.nn.init.xavier_normal_(hidden_state, gain=1.0)
        torch.nn.init.xavier_normal_(cell_state, gain=1.0)
        self.hidden = (hidden_state, cell_state)

    def forward(self, input_ids, labels, **kwargs):
        batch_size = input_ids.shape[0]
        self.init_hidden_states(batch_size)
        embed = self.embedding(input_ids)
        out, _ = self.lstm(embed, self.hidden)
        output = self.decoder(out)
        output = output.view(batch_size, self.target_seq_len_, self.tokenizer.vocab_size)
        loss = self.loss(output, labels)
        rnn_output = RNNOutput(logits=output, loss=loss)
        return rnn_output