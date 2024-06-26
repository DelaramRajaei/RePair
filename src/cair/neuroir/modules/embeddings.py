# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/embeddings.py
""" Embeddings module """
import math

import torch
import torch.nn as nn

from cair.neuroir.modules.util_class import Elementwise


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class CharEmbedding(nn.Module):
    """Embeds words based on character embeddings using CNN."""

    def __init__(self, vocab_size, emsize, filter_size, nfilters):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emsize)
        self.convolution = nn.ModuleList([nn.Conv1d(emsize, int(num_filter), int(k))
                                          for (k, num_filter) in zip(filter_size, nfilters)])

    def forward(self, inputs):
        """
        Embed words from character embeddings using CNN.
        Parameters
        --------------------
            inputs      -- 3d tensor (N,sentence_len,word_len)
        Returns
        --------------------
            loss        -- total loss over the input mini-batch (N,sentence_len,char_embed_size)
        """
        # step1: embed the characters
        char_emb = self.embedding(inputs.view(-1, inputs.size(2)))  # (N*sentence_len,word_len,char_emb_size)

        # step2: apply convolution to form word embeddings
        char_emb = char_emb.transpose(1, 2)  # (N*sentence_len,char_emb_size,word_len)
        output = []
        for conv in self.convolution:
            cnn_out = conv(char_emb).transpose(1, 2)  # (N*sentence_len,word_len-filter_size,num_filters)
            cnn_out = torch.max(cnn_out, 1)[0]  # (N*sentence_len,num_filters)
            output.append(cnn_out.view(*inputs.size()[:2], -1))  # appended (N,sentence_len,num_filters)

        output = torch.cat(output, 2)
        return output


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.
    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.
    .. mermaid::
       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]
    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 fix_word_vecs=False):

        if feat_padding_idx is None:
            feat_padding_idx = []

        self.word_vec_size = word_vec_size
        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]

        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

    @property
    def word_lut(self):
        """ word look-up table """
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """ embedding look-up table """
        return self.make_embedding[0]

    def expand_word_lut(self, new_vocab_size, emb_size, pad):
        old_embedding = self.word_lut.weight.data
        self.make_embedding[0][0] = torch.nn.Embedding(new_vocab_size,
                                                       emb_size,
                                                       padding_idx=pad)
        new_embedding = self.word_lut.weight.data
        new_embedding[:old_embedding.size(0)] = old_embedding

    def init_word_vectors(self, vocabulary, embeddings_index, fixed):
        """Initialize weight parameters for the word embedding layer.
        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        pretrained = torch.FloatTensor(len(vocabulary), self.word_vec_size).zero_()
        for i in range(len(vocabulary)):
            if vocabulary.ind2tok[i] in embeddings_index:
                pretrained[i] = embeddings_index[vocabulary.ind2tok[i]]

        self.word_lut.weight.data.copy_(pretrained)
        if fixed:
            self.word_lut.weight.requires_grad = False

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.
        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, source):
        """
        Computes the embeddings for words and features.
        Args:
            source (`LongTensor`): index tensor `[len x batch x nfeat]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        emb = self.make_embedding(source.transpose(0, 1))
        return emb.transpose(0, 1)
