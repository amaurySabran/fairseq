# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils
from fairseq.models.transformer import TransformerModel, TransformerEncoderLayer, TransformerDecoder, Embedding, \
    LayerNorm, PositionalEmbedding, Linear

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)


@register_model('transformer_conv')
class TransformerConvModel(TransformerModel):
    """
    A transformer model where a stridedconvolution is applied on the input sequence to reduce the
     input sequence size
    """

    def __init__(self, encoder, decoder):
        super(TransformerConvModel, self).__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(TransformerConvModel,TransformerConvModel).add_args(parser)
        parser.add_argument('--kernel-size', type=int, default=4)
        parser.add_argument('--deconv', default=False, action='store_true')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerConvEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerConvModel(encoder, decoder)


class TransformerConvEncoder(FairseqEncoder):
    """
    TransformerConv encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        self.conv_layer = nn.Conv1d(embed_dim, embed_dim, args.kernel_size, stride=args.kernel_size)
        self.kernel_size = args.kernel_size
        self.deconv = args.deconv

        if self.deconv:
            self.deconv_layer = nn.ConvTranspose1d(embed_dim, embed_dim, args.kernel_size, stride=args.kernel_size)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # padding
        batch_size, src_len, d = x.size()
        size_to_add = (self.kernel_size - src_len % self.kernel_size) % self.kernel_size
        src_tokens = F.pad(src_tokens, (size_to_add, 0), value=self.padding_idx)
        x = F.pad(x, (0, 0, size_to_add, 0))

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        x = self.conv_layer(x)

        # B x C x T -> B x T x C -> T x B x C
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        if self.deconv:
            encoder_padding_mask_ini = encoder_padding_mask.clone()

        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        else:
            # reshape the padding_mask
            # put a 1 in i if the 'kernel_size' elements that are merged to position i
            # by conv layer are all 1 
            encoder_padding_mask = encoder_padding_mask.view(-1, self.kernel_size)
            encoder_padding_mask = encoder_padding_mask.all(1)
            encoder_padding_mask = encoder_padding_mask.view(batch_size, -1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        if self.deconv:
            # T x B x C -> B x T x C -> B x C x T
            x = x.transpose(0, 1)
            x = x.transpose(1, 2)
            x = self.deconv_layer(x)

            # B x C x T -> B x T x C -> T x B x C
            x = x.transpose(1, 2)
            x = x.transpose(0, 1)

            # removing the parts that corresponded to the pad
            x = x[size_to_add:, :, :]
            encoder_padding_mask_ini = encoder_padding_mask_ini[:, size_to_add:]

            return {
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask_ini,  # B x T
            }

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict



@register_model_architecture('transformer_conv', 'transformer_conv')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.kernel_size = getattr(args, 'kernel_size', 4)
    args.deconv = getattr(args, 'deconv', False)


@register_model_architecture('transformer_conv', 'transformer_conv_small')
def transformer_conv_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    base_architecture(args)
