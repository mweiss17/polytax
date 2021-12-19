import torch
import torch.nn

class RNNModel(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_seq_len, target_seq_len_, batch_size, vocab_total_size):
        super(RNNModel, self).__init__()
        # input_seq_len: 32
        # target_seq_len: 8
        # batch_size: 4 #per device
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_total_size,
            embedding_dim=target_seq_len_)
        self.batch_size = batch_size
        self.vocab_total_size = vocab_total_size
        self.input_seq_len = input_seq_len
        self.target_seq_len_ = target_seq_len_
        self.lstm = torch.nn.LSTM(target_seq_len_, target_seq_len_, batch_first = True)

        self.second_decoder = torch.nn.Linear(target_seq_len_, vocab_total_size)
        self.first_decoder = torch.nn.Linear(batch_size * input_seq_len, target_seq_len_)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.act = torch.nn.Softplus(beta = 1, threshold = 20)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.embedding.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.first_decoder.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.second_decoder.weight, gain=1.0)
        self.first_decoder.bias.data.zero_()
        self.second_decoder.bias.data.zero_()

    def init_hidden_states(self):
        hidden1 = torch.zeros(1, self.batch_size, self.target_seq_len_)
        hidden2 = torch.zeros(1, self.batch_size, self.target_seq_len_)
        torch.nn.init.xavier_normal_(hidden1, gain=1.0)
        torch.nn.init.xavier_normal_(hidden2, gain=1.0)
        self.hidden = (hidden1,
          hidden2)

    def forward(self, input_ids, hidden = None, **kwargs):
        # TO DO: USE HIDDEN
        self.init_hidden_states()
        embed = self.embedding(input_ids)
        out, hidden = self.lstm(embed, self.hidden)
        decoder_out = torch.einsum("bst, td -> btd", out, self.first_decoder.weight)
        decoder_out_2 = torch.einsum("bsd, ms -> bms", decoder_out, self.second_decoder.weight)
        lstm_out = self.act(decoder_out_2)
        final_output = self.softmax(lstm_out)
        return final_output.view(self.batch_size, self.target_seq_len_, self.vocab_total_size), hidden