# coding: utf-8
import torch
import torch.nn.functional as F
import warnings
import numpy as np
import tqdm
from torch import nn
from torch.autograd import Variable

seed = 3636
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def noop(*args, **kwargs): return


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    res = Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)
    return res


class EmbeddingDropout(nn.Module):
    """ Applies dropout in the embedding layer by zeroing out some elements of the embedding vector.
    Uses the dropout_mask custom layer to achieve this.
    Args:
        embed (torch.nn.Embedding): An embedding torch layer
        words (torch.nn.Variable): A torch variable
        dropout (float): dropout fraction to apply to the embedding weights
        scale (float): additional scaling to apply to the modified embedding weights
    Returns:
        tensor of size: (batch_size x seq_length x embedding_size)
    """

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            size = (self.embed.weight.size(0), 1)
            mask = Variable(dropout_mask(self.embed.weight.data, size, dropout))  # define a mask to apply
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight

        if scale: masked_embed_weight = scale * masked_embed_weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None: padding_idx = -1

        X = self.embed._backend.Embedding.apply(words,
                                                masked_embed_weight, padding_idx, self.embed.max_norm,
                                                self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)

        return X


def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument 'sz'.
    Args:
        x (nn.Variable): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply
    This method uses the bernoulli distribution to decide which activations to keep.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).
    In the example given below, one can see that approximately .8 fraction of the
    returned tensors are zero. Rescaling with the factor 1/(1 - 0.8) returns a tensor
    with 5's in the unit places.
    """
    return x.new(*sz).bernoulli_(1 - dropout) / (1 - dropout)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(m, requires_grad=False) * x


class WeightDrop(torch.nn.Module):
    """A custom torch layer that serves as a wrapper on another torch layer.
    Primarily responsible for updating the weights in the wrapped module based
    on a specified dropout.
    """

    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        """ Default constructor for the WeightDrop module
        Args:
            module (torch.nn.Module): A pytorch layer being wrapped
            dropout (float): a dropout value to apply
            weights (list(str)): the parameters of the wrapped **module**
                which should be fractionally dropped.
        """
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        """ for each string defined in self.weights, the corresponding
        attribute in the wrapped module is referenced, then deleted, and subsequently
        registered as a new parameter with a slightly modified name.
        Args:
            None
         Returns:
             None
        """
        if isinstance(self.module, torch.nn.RNNBase): self.module.flatten_parameters = noop
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))


    def _setweights(self):
        """ Uses pytorch's built-in dropout function to apply dropout to the parameters of
        the wrapped module.
        Args:
            None
        Returns:
            None
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            if hasattr(self.module, name_w):
                delattr(self.module, name_w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        """ updates weights and delegates the propagation of the tensor to the wrapped module's
        forward method
        Args:
            *args: supplied arguments
        Returns:
            tensor obtained by running the forward method on the wrapped module.
        """
        self._setweights()
        return self.module.forward(*args)


class RNN_Encoder(nn.Module):
    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM
        - variational dropouts in the embedding and LSTM layers
        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange = 0.1

    def __init__(self, ntoken, emb_sz, n_hid, n_layers, pad_token, bidir=False,
                 dropout_hidden=0.3, dropout_input=0.65, dropout_embedding=0.1, dropout_weight=0.5):
        """ Default constructor for the RNN_Encoder class
            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                n_hid (int): number of hidden activation per LSTM layer
                n_layers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropout_embedding (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs = 1
        self.emb_sz = emb_sz,
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.dropout_embedding = dropout_embedding

        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)

        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz) // self.ndir,
                             1, bidirectional=bidir) for l in range(n_layers)]
        if dropout_weight: self.rnns = [WeightDrop(rnn, dropout_weight) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.reset_hidden()

        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.dropout_input = LockedDropout(dropout_input)
        self.dropouths = nn.ModuleList([LockedDropout(dropout_hidden) for l in range(n_layers)])

    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)
        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset_hidden()

        emb = self.encoder_with_dropout(input, dropout=self.dropout_embedding if self.training else 0)
        # emb                                 = self.dropout_input(emb)
        raw_output = emb
        new_hidden, raw_outputs, outputs = [], [], []
        for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = drop(raw_output)
            outputs.append(raw_output)

            self.hidden[l] = repackage_var(new_h)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz[0]) // self.ndir
        return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset_hidden(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.n_layers)]


class LinearDecoder(nn.Module):
    initrange = 0.1

    def __init__(self, n_out, n_hid, dropout, tie_encoder=None, bias=False):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


def get_language_model(n_tok, emb_sz, nhid, nlayers, pad_token, decode_train=True, dropouts=None):
    if dropouts is None: dropouts = [0.5, 0.4, 0.5, 0.05, 0.3]

    rnn_enc = RNN_Encoder(n_tok, emb_sz, n_hid=nhid, bidir=False, n_layers=nlayers, pad_token=pad_token,
                          dropout_input=dropouts[0], dropout_weight=dropouts[2], dropout_embedding=dropouts[3],
                          dropout_hidden=dropouts[4])
    rnn_dec = LinearDecoder(n_tok, emb_sz, 0.2, tie_encoder=rnn_enc.encoder, bias=False)
    #
    # rnn_dec = LinearDecoder(n_tok, em_sz, dropouts[1], decode_train=decode_train, tie_encoder=rnn_enc.encoder)

    return nn.Sequential(rnn_enc, rnn_dec)


class LanguageModelLoader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """

    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        self.data = self.batchify(nums)
        self.i = 0
        self.iter = 0,
        self.n = len(self.data)

    def __iter__(self):
        self.i = 0
        self.iter = 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 25
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb * self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data = data[::-1]
        return torch.LongTensor(data).cuda()

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i + seq_len], source[i + 1:i + 1 + seq_len].view(-1)
