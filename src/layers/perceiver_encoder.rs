use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

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

        todo!()
    }

    pub fn forward(&self, x: &Tensor, mask: Option<Tensor>) -> Result<Tensor> {
        if let Some(proj_context) = self.proj_context.as_ref() {
            let x = proj_context.forward(x)?;
        }

        todo!()
    }
}

pub struct AttentionConfig {
    pub dim: usize,
    pub dim_context: usize,
    pub causal: bool,
    pub dim_head: usize,
    pub heads: usize,
    pub dropout: f64,
    pub use_flash: bool,
    pub cross_attn_include_queries: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            dim: 512,
            dim_context: 512,
            causal: false,
            dim_head: 64,
            heads: 8,
            dropout: 0.0,
            use_flash: false,
            cross_attn_include_queries: false,
        }
    }
}

pub struct Attention {}

impl Attention {
    pub fn new(vb: VarBuilder, config: &AttentionConfig) -> Result<Self> {
        todo!()
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<Tensor>,
        mask: Option<Tensor>,
    ) -> Result<Tensor> {
        todo!()
    }
}
