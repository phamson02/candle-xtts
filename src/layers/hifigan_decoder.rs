use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::hifigan_generator::{ResBlock, ResBlockType};

pub struct SpeakerEncoderAudioConfig {
    fft_size: usize,
    win_length: usize,
    hop_length: usize,
    sample_rate: usize,
    preemphasis: f64,
    num_mels: usize,
}

impl Default for SpeakerEncoderAudioConfig {
    fn default() -> Self {
        Self {
            fft_size: 512,
            win_length: 400,
            hop_length: 160,
            sample_rate: 16000,
            preemphasis: 0.97,
            num_mels: 64,
        }
    }
}

pub struct HifiDecoderConfig {
    input_sample_rate: usize,
    output_sample_rate: usize,
    output_hop_length: usize,
    ar_mel_length_compression: usize,
    decoder_input_dim: usize,
    resblock_type_decoder: ResBlockType,
    resblock_dilation_sizes_decoder: Vec<Vec<usize>>,
    upsample_rates_decoder: Vec<usize>,
    upsample_initial_channel_decoder: usize,
    upsample_kernel_sizes_decoder: Vec<usize>,
    d_vector_dim: usize,
    cond_d_vector_in_each_upsampling_layer: bool,
    speaker_encoder_audio_config: SpeakerEncoderAudioConfig,
}

impl Default for HifiDecoderConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: 22050,
            output_sample_rate: 24000,
            output_hop_length: 256,
            ar_mel_length_compression: 1024,
            decoder_input_dim: 1024,
            resblock_type_decoder: ResBlockType::ResBlock1,
            resblock_dilation_sizes_decoder: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_rates_decoder: vec![8, 8, 2, 2],
            upsample_initial_channel_decoder: 512,
            upsample_kernel_sizes_decoder: vec![16, 16, 4, 4],
            d_vector_dim: 512,
            cond_d_vector_in_each_upsampling_layer: true,
            speaker_encoder_audio_config: SpeakerEncoderAudioConfig::default(),
        }
    }
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
