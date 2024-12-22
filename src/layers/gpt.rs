use candle_core::Result;
use candle_nn::{embedding, VarBuilder};

pub struct GPTConfig {
    start_text_token: usize,
    stop_text_token: usize,
    layers: usize,
    model_dim: usize,
    heads: usize,
    max_text_tokens: usize,
    max_mel_tokens: usize,
    max_prompt_tokens: usize,
    max_conditioning_inputs: usize,
    code_stride_len: usize,
    number_text_tokens: usize,
    num_audio_tokens: usize,
    start_audio_token: usize,
    stop_audio_token: usize,
    train_solo_embeddings: bool,
    checkpointing: bool,
    average_conditioning_embeddings: bool,
    label_smoothing: f64,
    use_perceiver_resampler: bool,
    perceiver_cond_length_compression: usize,
}

pub struct GPT {}

impl GPT {
    pub fn load(vb: VarBuilder, config: &GPTConfig) -> Result<Self> {
        let text_embedding = embedding(
            config.number_text_tokens,
            config.model_dim,
            vb.pp("text_embedding"),
        )?;
        let mel_embedding = embedding(
            config.num_audio_tokens,
            config.model_dim,
            vb.pp("mel_embedding"),
        );

        todo!()
    }
}
