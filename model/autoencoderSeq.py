from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _get_padding_mask, _get_key_padding_mask


class EmbeddingSeq(nn.Module):
    """Embedding: positional embed + operation embed + zs embed"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.operation_encoding = nn.Embedding(cfg.n_operations, cfg.dim_z)

        self.zs_encoding = nn.Linear(cfg.dim_z, cfg.dim_z)

        self.pos_encoding = PositionalEncodingLUT(cfg.dim_z, max_len=seq_len)

    def forward(self, seq_op, seq_zs):
        # seq_op: (S, N), seq_zs: (S, N, dim_z)

        src = self.operation_encoding(seq_op.long()) + self.zs_encoding(seq_zs.float()) # (S, N, dim_z)

        src = self.pos_encoding(src) # (S, N, dim_z)

        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.embedding = EmbeddingSeq(cfg, seq_len)

        encoder_layer = TransformerEncoderLayerImproved(cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.dim_z)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, seq_op, seq_zs):
        padding_mask, key_padding_mask = _get_padding_mask(seq_op, seq_dim=0), _get_key_padding_mask(seq_op, seq_dim=0)

        src = self.embedding(seq_op, seq_zs) # (S, N, dim_z)

        memory = self.encoder(src, src_attn_mask=None, src_key_padding_mask=key_padding_mask) # (S, N, dim_z)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.dim_z, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z) # (1, N, dim_z)


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.dim_z = cfg.dim_z
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.dim_z, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.dim_z)) # (S, N, dim_z)
        return src


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.dim_z, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.dim_z)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        self.operation_decode =  nn.Linear(cfg.dim_z, cfg.n_operations)
        self.zs_decode = nn.Linear(cfg.dim_z, cfg.dim_z)

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_attn_mask=None, tgt_key_padding_mask=None) # (S, N, dim_z)

        seq_op_dec = self.operation_decode(out)  # Shape [S, N, n_operations]
        seq_zs_dec = self.zs_decode(out)  # Shape [S, N, dim_z]

        out_tuple = (seq_op_dec, seq_zs_dec)
        return out_tuple


class AutoEncoderSeq(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderSeq, self).__init__()

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)

    def forward(self, seq_op, seq_zs,
                z=None, return_tgt=True, encode_mode=False):

        if z is None:
            z = self.encoder(seq_op, seq_zs)
            z = self.bottleneck(z)
        else:
            z = torch.unsqueeze(z, 0) # Shape [1, N, dim_z]

        if encode_mode: return torch.squeeze(z, 0) # Shape [N, dim_z]

        out_tuple = self.decoder(z)

        res = {
            "seq_op_out": out_tuple[0], # Shape [S, N, n_operations]
            "seq_zs_out": out_tuple[1] # Shape [S, N, dim_z]
        }

        if return_tgt:
            res["seq_op_tgt"] = seq_op # Shape [S, N]
            res["seq_zs_tgt"] = seq_zs # Shape [S, N, dim_z]

        return res
