use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct SpeakerEncoderAudioConfig {
    fft_size: usize,
    win_length: usize,
    hop_length: usize,
    sample_rate: usize,
    preemphasis: f64,
    num_mels: usize,
}

pub struct HifiDecoderConfig {
    input_sample_rate: usize,
    output_sample_rate: usize,
    output_hop_length: usize,
    ar_mel_length_compression: usize,
    decoder_input_dim: usize,
    resblock_type_decoder: String,
    resblock_dilation_sizes_decoder: Vec<Vec<usize>>,
    upsample_rates_decoder: Vec<usize>,
    upsample_initial_channel_decoder: usize,
    upsample_kernel_sizes_decoder: Vec<usize>,
    d_vector_dim: usize,
    cond_d_vector_in_each_upsampling_layer: bool,
    speaker_encoder_audio_config: SpeakerEncoderAudioConfig,
}

pub struct HifiDecoder {}

impl HifiDecoder {
    pub fn load(vb: VarBuilder, config: &HifiDecoderConfig) -> Result<Self> {
        todo!()
    }

    pub fn forward(&self, latents: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        todo!()
    }
}
