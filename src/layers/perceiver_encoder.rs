use candle_core::{Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, ops::softmax, Linear, Module, VarBuilder};

pub struct GEGLU {}

impl GEGLU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match (x.chunk(2, D::Minus1)?).as_slice() {
            [x, gates] => x * gates.gelu()?,
            _ => unreachable!(),
        }
    }
}

pub struct FeedForward {
    linear1: Linear,
    geglu: GEGLU,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let dim_inner = dim * mult * 2 / 3;
        let linear1 = linear(dim, dim_inner * 2, vb.pp("linear1"))?;
        let linear2 = linear(dim_inner, dim, vb.pp("linear2"))?;
        let geglu = GEGLU {};
        Ok(Self {
            linear1,
            geglu,
            linear2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.geglu.forward(&x)?;
        self.linear2.forward(&x)
    }
}

pub struct AttentionBlock {
    attention: Attention,
    feed_forward: FeedForward,
}

impl AttentionBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        dim_head: usize,
        heads: usize,
        mult: usize,
    ) -> Result<Self> {
        let attention = Attention::new(
            vb.pp("attention"),
            &AttentionConfig {
                dim,
                dim_context: dim,
                dim_head,
                heads,
            },
        )?;
        let feed_forward = FeedForward::new(vb.pp("ff"), dim, mult)?;
        Ok(Self {
            attention,
            feed_forward,
        })
    }
}

pub struct PerceiverResamplerConfig {
    pub dim: usize,
    pub depth: usize,
    pub dim_context: usize,
    pub num_latents: usize,
    pub dim_head: usize,
    pub heads: usize,
    pub ff_mult: usize,
    pub use_flash_attn: bool,
}

impl Default for PerceiverResamplerConfig {
    fn default() -> Self {
        Self {
            dim: 512,
            depth: 2,
            dim_context: 512,
            num_latents: 32,
            dim_head: 64,
            heads: 8,
            ff_mult: 4,
            use_flash_attn: false,
        }
    }
}

pub struct PerceiverResampler {
    proj_context: Option<Linear>,
    layers: Vec<AttentionBlock>,
    latents: Tensor,
}

impl PerceiverResampler {
    pub fn new(vb: VarBuilder, config: &PerceiverResamplerConfig) -> Result<Self> {
        let proj_context = if config.dim_context != config.dim {
            Some(linear(
                config.dim_context,
                config.dim,
                vb.pp("proj_context"),
            )?)
        } else {
            None
        };

        let latents = vb.get((config.num_latents, config.dim), "latents")?;

        let mut layers = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            layers.push(AttentionBlock::new(
                vb.pp(&format!("layer{}", i)),
                config.dim,
                config.dim_head,
                config.heads,
                config.ff_mult,
            )?);
        }

        todo!()
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let batch = x.dims()[0];
        if let Some(proj_context) = self.proj_context.as_ref() {
            let x = proj_context.forward(x)?;
        }

        let latents =
            self.latents
                .repeat((batch, self.latents.dims()[0], self.latents.dims()[1]))?;

        for layer in &self.layers {
            let latents = (layer.attention.forward(&latents.clone(), x, mask)? + latents.clone())?;
            let latents = (layer.feed_forward.forward(&latents.clone())? + latents.clone())?;
        }

        todo!()
    }
}

pub struct AttentionConfig {
    pub dim: usize,
    pub dim_context: usize,
    pub dim_head: usize,
    pub heads: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            dim: 512,
            dim_context: 512,
            dim_head: 64,
            heads: 8,
        }
    }
}

pub struct Attention {
    heads: usize,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    attend: Attend,
}

impl Attention {
    pub fn new(vb: VarBuilder, config: &AttentionConfig) -> Result<Self> {
        let dim_inner = config.dim_head * config.heads;
        let to_q = linear_no_bias(config.dim, dim_inner, vb.pp("to_q"))?;
        let to_kv = linear_no_bias(config.dim_context, dim_inner * 2, vb.pp("to_kv"))?;
        let to_out = linear_no_bias(dim_inner, config.dim, vb.pp("to_out"))?;
        let attend = Attend {};

        Ok(Self {
            heads: config.heads,
            to_q,
            to_kv,
            to_out,
            attend,
        })
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.heads, n_state / self.heads];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let context = Tensor::cat(&[x, context], D::Minus2)?;

        let q = self.to_q.forward(x)?;
        let kv = self.to_kv.forward(&context)?.chunk(2, D::Minus1)?;

        match kv.as_slice() {
            [k, v] => {
                let q = self.reshape_head(&q)?;
                let k = self.reshape_head(&k)?;
                let v = self.reshape_head(&v)?;
                let out = self
                    .attend
                    .forward(&q, &k, &v, mask)?
                    .transpose(1, 2)?
                    .flatten_from(2)?;
                Ok(self.to_out.forward(&out)?)
            }
            _ => unreachable!(),
        }
    }
}

pub struct Attend {}

impl Attend {
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, _, _, n_dim) = q.dims4()?;
        let scale = (n_dim as f64).powf(-0.25);
        let q = (q * scale)?;
        let k = (k.transpose(2, 3)? * scale)?;
        let v = v.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = { softmax(&qk, D::Minus1)? };
        let wv = w.matmul(&v)?;
        Ok(wv)
    }
}
