use candle_core::{Result, Tensor};
use candle_nn::{
    conv1d, group_norm, seq, Conv1d, Conv1dConfig, GroupNorm, Module, Sequential, VarBuilder,
};

fn normalization(vb: VarBuilder, channels: usize) -> Result<GroupNorm> {
    let mut groups = 32;
    if channels <= 16 {
        groups = 8;
    } else if channels <= 64 {
        groups = 16;
    }
    if channels % groups != 0 {
        groups /= 2;
    }
    if groups <= 2 {
        candle_core::bail!("GroupNorm: num_groups ({}) must be greater than 2", groups);
    }
    group_norm(groups, channels, 1e-05, vb)
}

pub struct QKVAttentionLegacy {
    pub n_heads: usize,
}

impl Module for QKVAttentionLegacy {
    fn forward(&self, qkv: &Tensor) -> Result<Tensor> {
        let (bs, width, length) = qkv.dims3()?;
        let ch = width / (self.n_heads * 3);
        let qkv = qkv
            .reshape((bs * self.n_heads, ch * 3, length))?
            .chunk(ch, 1)?;

        match qkv.as_slice() {
            [q, k, v] => {
                let scale = 1.0 / (ch as f64).powf(-0.25);
                todo!()
            }
            _ => candle_core::bail!("Expected 3 chunks"),
        }
    }
}

pub struct AttentionBlockConfig {
    pub channels: usize,
    pub num_heads: usize,
}

pub struct AttentionBlock {
    norm: GroupNorm,
    qkv: Conv1d,
    attention: QKVAttentionLegacy,
    proj_out: Conv1d,
}

impl AttentionBlock {
    pub fn new(vb: VarBuilder, config: &AttentionBlockConfig) -> Result<Self> {
        let norm = normalization(vb.pp("norm"), config.channels)?;
        let qkv = conv1d(
            config.channels,
            config.channels * 3,
            1,
            Conv1dConfig {
                ..Conv1dConfig::default()
            },
            vb.pp("qkv"),
        )?;
        let attention = QKVAttentionLegacy {
            n_heads: config.num_heads,
        };
        let proj_out = conv1d(
            config.channels,
            config.channels,
            1,
            Conv1dConfig {
                ..Conv1dConfig::default()
            },
            vb.pp("proj_out"),
        )?;

        Ok(Self {
            norm,
            qkv,
            attention,
            proj_out,
        })
    }
}

impl Module for AttentionBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.shape().dim(0)?;
        let c = x.shape().dim(1)?;
        let spatial = x.shape().dims().get(2..).unwrap();

        let x = x.reshape((b, c, ()))?;
        let x_norm = x.apply(&self.norm)?;
        let qkv = x_norm.apply(&self.qkv)?;
        let h = qkv.apply(&self.attention)?;
        let h = h.apply(&self.proj_out)?;

        let mut new_shape = vec![b, c];
        new_shape.extend(spatial);
        (x_norm + h)?.reshape(new_shape)
    }
}

pub struct ConditioningEncoderConfig {
    pub spec_dim: usize,
    pub embedding_dim: usize,
    pub attn_blocks: usize,
    pub num_attn_heads: usize,
}

impl ConditioningEncoderConfig {
    pub fn default(spec_dim: usize, embedding_dim: usize) -> Self {
        Self {
            spec_dim,
            embedding_dim,
            attn_blocks: 6,
            num_attn_heads: 4,
        }
    }
}

pub struct ConditioningEncoder {
    init: Conv1d,
    attn: Sequential,
}

impl ConditioningEncoder {
    pub fn new(vb: VarBuilder, config: &ConditioningEncoderConfig) -> Result<Self> {
        let init = conv1d(
            config.spec_dim,
            config.embedding_dim,
            1,
            Conv1dConfig {
                ..Conv1dConfig::default()
            },
            vb.pp("init"),
        )?;

        let mut attn = seq();
        for i in 0..config.attn_blocks {
            attn = attn.add(AttentionBlock::new(
                vb.pp(format!("attn.{}", i)),
                &AttentionBlockConfig {
                    channels: config.embedding_dim,
                    num_heads: config.num_attn_heads,
                },
            )?);
        }

        Ok(Self { init, attn })
    }
}

impl Module for ConditioningEncoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.init)?.apply(&self.attn)
    }
}
