use crate::layers::{
    conditioning_encoder::ConditioningEncoderConfig, perceiver_encoder::PerceiverResamplerConfig,
};

use super::{conditioning_encoder::ConditioningEncoder, perceiver_encoder::PerceiverResampler};
use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, Embedding, LayerNorm, LayerNormConfig, Linear, Module,
    VarBuilder,
};

fn gpt2_linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let ws = vb.get((in_dim, out_dim), "weight")?;
    let ws = ws.t()?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

struct LearnedPositionEmbeddings {
    emb: Embedding,
    seq_len: usize,
}

impl LearnedPositionEmbeddings {
    pub fn new(vb: VarBuilder, seq_len: usize, model_dim: usize) -> Result<Self> {
        let emb = embedding(seq_len, model_dim, vb.pp("emb"))?;
        Ok(Self { emb, seq_len })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sl = x.shape().dim(1)?;
        self.emb
            .forward(&Tensor::arange(0u32, sl as u32, x.device())?)
    }
}

#[derive(Debug, Clone)]
pub struct GPT2LayerConfig {
    pub n_head: usize,
    pub n_embd: usize,
}

struct SelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
}

impl SelfAttention {
    fn new(cfg: &GPT2LayerConfig, vb: VarBuilder) -> Result<Self> {
        let c_attn = gpt2_linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?;
        let c_proj = gpt2_linear(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
        })
    }
}

impl Module for SelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, c) = xs.dims3()?;
        let c_x = xs
            .apply(&self.c_attn)?
            .reshape((b, t, 3, self.n_head, c / self.n_head))?;
        let q = c_x.i((.., .., 0))?;
        let k = c_x.i((.., .., 1))?;
        let v = c_x.i((.., .., 2))?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let att = (q.matmul(&k.t()?)? / (k.dim(D::Minus1)? as f64).sqrt())?;
        // TODO: causal mask
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let att = att.matmul(&v)?.transpose(1, 2)?;
        att.reshape((b, t, c))?.apply(&self.c_proj)
    }
}

pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn new(cfg: &GPT2LayerConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = 4 * cfg.n_embd;
        let c_fc = gpt2_linear(cfg.n_embd, hidden_dim, vb.pp("c_fc"))?;
        let c_proj = gpt2_linear(hidden_dim, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.c_fc)?.gelu()?.apply(&self.c_proj)
    }
}

struct GPT2Layer {
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    attn: SelfAttention,
    mlp: MLP,
}

impl GPT2Layer {
    fn new(cfg: &GPT2LayerConfig, vb: VarBuilder) -> Result<Self> {
        let ln_1 = layer_norm(
            cfg.n_embd,
            LayerNormConfig {
                ..Default::default()
            },
            vb.pp("ln_1"),
        )?;
        let ln_2 = layer_norm(
            cfg.n_embd,
            LayerNormConfig {
                ..Default::default()
            },
            vb.pp("ln_2"),
        )?;
        let attn = SelfAttention::new(cfg, vb.pp("attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(GPT2Layer {
            ln_1,
            ln_2,
            attn,
            mlp,
        })
    }
}

impl Module for GPT2Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + xs.apply(&self.ln_1)?.apply(&self.attn))?;
        let xs = (&xs + xs.apply(&self.ln_2)?.apply(&self.mlp))?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub start_text_token: usize,
    pub stop_text_token: usize,
    pub layers: usize,
    pub model_dim: usize,
    pub heads: usize,
    pub max_text_tokens: usize,
    pub max_mel_tokens: usize,
    pub max_prompt_tokens: usize,
    pub max_conditioning_inputs: usize,
    pub code_stride_len: usize,
    pub number_text_tokens: usize,
    pub num_audio_tokens: usize,
    pub start_audio_token: usize,
    pub stop_audio_token: usize,
    pub train_solo_embeddings: bool,
    pub checkpointing: bool,
    pub average_conditioning_embeddings: bool,
    pub label_smoothing: f64,
    pub use_perceiver_resampler: bool,
    pub perceiver_cond_length_compression: usize,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            start_text_token: 261,
            stop_text_token: 0,
            layers: 8,
            model_dim: 512,
            heads: 8,
            max_text_tokens: 129,
            max_mel_tokens: 250,
            max_prompt_tokens: 70,
            max_conditioning_inputs: 1,
            code_stride_len: 1024,
            number_text_tokens: 256,
            num_audio_tokens: 8194,
            start_audio_token: 8192,
            stop_audio_token: 8193,
            train_solo_embeddings: false,
            checkpointing: false,
            average_conditioning_embeddings: false,
            label_smoothing: 0.0,
            use_perceiver_resampler: true,
            perceiver_cond_length_compression: 256,
        }
    }
}

pub struct GPT {
    config: GPTConfig,
    stop_audio_token: usize,
    text_embedding: Embedding,
    mel_embedding: Embedding,
    text_pos_embedding: LearnedPositionEmbeddings,
    mel_pos_embedding: LearnedPositionEmbeddings,
    gpt_layers: Vec<GPT2Layer>,
    final_norm: LayerNorm,
    text_head: Linear,
    mel_head: Linear,
    conditioning_perceiver: PerceiverResampler,
    conditioning_encoder: ConditioningEncoder,
}

impl GPT {
    pub fn new(vb: VarBuilder, config: &GPTConfig) -> Result<Self> {
        let max_mel_tokens = config.max_mel_tokens + 2 + config.max_conditioning_inputs;
        let max_text_tokens = config.max_text_tokens + 2;

        let conditioning_encoder = ConditioningEncoder::new(
            vb.pp("conditioning_encoder"),
            &ConditioningEncoderConfig {
                spec_dim: 80,
                embedding_dim: config.model_dim,
                attn_blocks: 6,
                num_attn_heads: config.heads,
            },
        )?;

        let text_embedding = embedding(
            config.number_text_tokens,
            config.model_dim,
            vb.pp("text_embedding"),
        )?;
        let mel_embedding = embedding(
            config.num_audio_tokens,
            config.model_dim,
            vb.pp("mel_embedding"),
        )?;

        let mel_pos_embedding = LearnedPositionEmbeddings::new(
            vb.pp("mel_pos_embedding"),
            max_mel_tokens,
            config.model_dim,
        )?;

        let text_pos_embedding = LearnedPositionEmbeddings::new(
            vb.pp("text_pos_embedding"),
            max_text_tokens,
            config.model_dim,
        )?;

        let mut gpt_layers: Vec<GPT2Layer> = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            gpt_layers.push(GPT2Layer::new(
                &GPT2LayerConfig {
                    n_head: config.heads,
                    n_embd: config.model_dim,
                },
                vb.pp(format!("gpt.h.{}", i)),
            )?);
        }

        let conditioning_perceiver = PerceiverResampler::new(
            vb.pp("conditioning_perceiver"),
            &PerceiverResamplerConfig {
                dim: config.model_dim,
                depth: 2,
                dim_context: config.model_dim,
                num_latents: 32,
                dim_head: 64,
                heads: 8,
                ff_mult: 4,
            },
        )?;

        let final_norm = layer_norm(
            config.model_dim,
            LayerNormConfig {
                ..Default::default()
            },
            vb.pp("final_norm"),
        )?;

        let text_head = linear(
            config.model_dim,
            config.number_text_tokens,
            vb.pp("text_head"),
        )?;

        let mel_head = linear(config.model_dim, config.num_audio_tokens, vb.pp("mel_head"))?;

        Ok(Self {
            config: config.clone(),
            stop_audio_token: config.stop_audio_token,
            text_embedding,
            mel_embedding,
            text_pos_embedding,
            mel_pos_embedding,
            gpt_layers,
            final_norm,
            text_head,
            mel_head,
            conditioning_perceiver,
            conditioning_encoder,
        })
    }

    fn compute_embeddings(&self, cond_latents: &Tensor, text_inputs: &Tensor) -> Result<Tensor> {
        // text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        // text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)

        let emb = (self.text_embedding.forward(&text_inputs)?
            + self.text_pos_embedding.forward(&text_inputs)?)?;
        let emb = Tensor::cat(&[cond_latents, &emb], 1)?;

        // self.gpt_inference.store_prefix_emb(emb)

        let gpt_inputs = Tensor::ones(
            (emb.shape().dim(0)?, emb.shape().dim(1)? + 1),
            candle_core::DType::I64,
            text_inputs.device(),
        )?;

        // gpt_inputs[:, -1] = self.start_audio_token

        Ok(gpt_inputs)
    }

    pub fn generate(&self, cond_latents: &Tensor, text_inputs: &Tensor) -> Result<Tensor> {
        let gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)?;
        let stop_token_tensor = Tensor::from_vec(
            vec![self.stop_audio_token as i64],
            (1, 1),
            text_inputs.device(),
        )?;
        let attention_mask = Tensor::ones(
            (gpt_inputs.shape().dim(0)?, gpt_inputs.shape().dim(1)?),
            candle_core::DType::I64,
            text_inputs.device(),
        )?;

        todo!()
    }

    pub fn get_style_emb(&self, cond_input: &Tensor, return_latent: bool) -> Result<Tensor> {
        let conds = if !return_latent {
            let cond_input = if cond_input.dims().len() == 4 {
                cond_input.squeeze(1)?
            } else {
                cond_input.clone()
            };
            let conds = self.conditioning_encoder.forward(&cond_input)?;
            let conds = if self.config.use_perceiver_resampler {
                self.conditioning_perceiver
                    .forward(&conds.permute((0, 2, 1))?, None)?
                    .transpose(1, 2)?
            } else {
                conds
            };
            conds
        } else {
            cond_input.unsqueeze(1)?
        };
        Ok(conds)
    }

    pub fn forward(
        &self,
        text_inputs: &Tensor,
        text_lengths: &Tensor,
        audio_codes: &Tensor,
        wav_lengths: &Tensor,
        cond_models: &Tensor,
    ) -> Result<Tensor> {
        let text_emb = (self.text_embedding.forward(&text_inputs)?
            + self.text_pos_embedding.forward(&text_inputs)?)?;
        let mel_emb = (self.mel_embedding.forward(&audio_codes)?
            + self.mel_pos_embedding.forward(&audio_codes)?)?;

        todo!()
    }
}
