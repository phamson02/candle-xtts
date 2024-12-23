use candle_core::Result;
use candle_nn::{embedding, VarBuilder};

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

pub struct GPT {}

impl GPT {
    pub fn new(vb: VarBuilder, config: &GPTConfig) -> Result<Self> {
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

        todo!()
    }
}
