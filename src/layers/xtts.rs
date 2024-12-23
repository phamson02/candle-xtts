use candle_core::Result;
use candle_nn::VarBuilder;

use crate::layers::{
    gpt::{GPTConfig, GPT},
    hifigan_decoder::{HifiDecoder, HifiDecoderConfig},
    tokenizer::{self, VoiceBpeTokenizer},
};

pub struct XTTSConfig {
    // A dataclass to represent XTTS model arguments that define the model structure.

    // Args:
    //     gpt_batch_size (int): The size of the auto-regressive batch.
    //     enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
    //     kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
    //     gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
    //     clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
    //     decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
    //     num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

    //     For GPT model:
    //     gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
    //     gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
    //     gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
    //     gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
    //     gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
    //     gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
    //     gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
    //     gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
    //     gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
    //     gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
    //     gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
    //     gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
    //     gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    gpt_batch_size: usize,
    enable_redaction: bool,
    kv_cache: bool,
    num_chars: usize,

    // XTTS GPT Encoder params
    gpt_max_audio_tokens: usize,
    gpt_max_text_tokens: usize,
    gpt_max_prompt_tokens: usize,
    gpt_layers: usize,
    gpt_n_model_channels: usize,
    gpt_n_heads: usize,
    gpt_number_text_tokens: usize,
    gpt_num_audio_tokens: usize,
    gpt_start_audio_token: usize,
    gpt_stop_audio_token: usize,
    gpt_code_stride_len: usize,
    gpt_use_masking_gt_prompt_approach: bool,
    gpt_use_perceiver_resampler: bool,

    // HifiGAN Decoder params
    input_sample_rate: usize,
    output_sample_rate: usize,
    output_hop_length: usize,
    decoder_input_dim: usize,
    d_vector_dim: usize,
    cond_d_vector_in_each_upsampling_layer: bool,

    // constants
    duration_const: usize,
}

impl Default for XTTSConfig {
    fn default() -> Self {
        Self {
            gpt_batch_size: 1,
            enable_redaction: false,
            kv_cache: true,
            num_chars: 255,
            gpt_max_audio_tokens: 605,
            gpt_max_text_tokens: 402,
            gpt_max_prompt_tokens: 70,
            gpt_layers: 30,
            gpt_n_model_channels: 1024,
            gpt_n_heads: 16,
            gpt_number_text_tokens: 6681,
            gpt_num_audio_tokens: 1026,
            gpt_start_audio_token: 1024,
            gpt_stop_audio_token: 1025,
            gpt_code_stride_len: 1024,
            gpt_use_masking_gt_prompt_approach: true,
            gpt_use_perceiver_resampler: true,
            input_sample_rate: 22050,
            output_sample_rate: 24000,
            output_hop_length: 256,
            decoder_input_dim: 1024,
            d_vector_dim: 512,
            cond_d_vector_in_each_upsampling_layer: true,
            duration_const: 102400,
        }
    }
}

pub struct XTTS {
    tokenizer: VoiceBpeTokenizer,
    gpt: GPT,
    hifigan_decoder: HifiDecoder,
}

impl XTTS {
    pub fn new(vb: VarBuilder, config: &XTTSConfig) -> Result<Self> {
        let tokenizer = VoiceBpeTokenizer::new();

        // if self.args.gpt_number_text_tokens -> True
        let gpt = GPT::new(
            vb.pp("gpt"),
            &GPTConfig {
                layers: config.gpt_layers,
                model_dim: config.gpt_n_model_channels,
                start_text_token: config.gpt_start_audio_token,
                stop_text_token: config.gpt_stop_audio_token,
                heads: config.gpt_n_heads,
                max_text_tokens: config.gpt_max_text_tokens,
                max_mel_tokens: config.gpt_max_audio_tokens,
                max_prompt_tokens: config.gpt_max_prompt_tokens,
                number_text_tokens: config.gpt_number_text_tokens,
                num_audio_tokens: config.gpt_num_audio_tokens,
                start_audio_token: config.gpt_start_audio_token,
                stop_audio_token: config.gpt_stop_audio_token,
                use_perceiver_resampler: config.gpt_use_perceiver_resampler,
                code_stride_len: config.gpt_code_stride_len,
                ..GPTConfig::default()
            },
        )?;

        let hifigan_decoder = HifiDecoder::new(
            vb.pp("hifigan_decoder"),
            &HifiDecoderConfig {
                input_sample_rate: config.input_sample_rate,
                output_sample_rate: config.output_sample_rate,
                ar_mel_length_compression: config.duration_const,
                decoder_input_dim: config.decoder_input_dim,
                d_vector_dim: config.d_vector_dim,
                cond_d_vector_in_each_upsampling_layer: config
                    .cond_d_vector_in_each_upsampling_layer,
                ..HifiDecoderConfig::default()
            },
        )?;

        todo!()
    }
}
