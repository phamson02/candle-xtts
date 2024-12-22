use candle_core::Result;
use candle_nn::VarBuilder;

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
    gpt_checkpoint: String,
    clvp_checkpoint: String,
    decoder_checkpoint: String,
    num_chars: usize,

    // XTTS GPT Encoder params
    tokenizer_file: String,
    gpt_max_audio_tokens: usize,
    gpt_max_text_tokens: usize,
    gpt_max_prompt_tokens: usize,
    gpt_layers: usize,
    gpt_n_model_channels: usize,
    gpt_n_heads: usize,
    gpt_number_text_tokens: usize,
    gpt_start_text_token: usize,
    gpt_stop_text_token: usize,
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

pub struct XTTS {}

impl XTTS {
    pub fn load(vb: VarBuilder, config: &XTTSConfig) -> Result<Self> {
        todo!()
    }
}
