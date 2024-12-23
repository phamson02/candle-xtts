use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::{
    hifigan_generator::{HifiganGenerator, HifiganGeneratorConfig},
    resnet::{ResNetSpeakerEncoder, ResNetSpeakerEncoderConfig},
};

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
pub struct HifiDecoderConfig {
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub output_hop_length: usize,
    pub ar_mel_length_compression: usize,
    pub decoder_input_dim: usize,
    pub d_vector_dim: usize,
    pub cond_d_vector_in_each_upsampling_layer: bool,
    pub speaker_encoder_audio_config: SpeakerEncoderAudioConfig,
}

impl Default for HifiDecoderConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: 22050,
            output_sample_rate: 24000,
            output_hop_length: 256,
            ar_mel_length_compression: 1024,
            decoder_input_dim: 1024,
            d_vector_dim: 512,
            cond_d_vector_in_each_upsampling_layer: true,
            speaker_encoder_audio_config: SpeakerEncoderAudioConfig::default(),
        }
    }
}

pub struct HifiDecoder {
    config: HifiDecoderConfig,
    waveform_decoder: HifiganGenerator,
    speaker_encoder: ResNetSpeakerEncoder,
}

impl HifiDecoder {
    pub fn new(vb: VarBuilder, config: &HifiDecoderConfig) -> Result<Self> {
        let waveform_decoder = HifiganGenerator::new(
            vb.pp("waveform_decoder"),
            &HifiganGeneratorConfig::default(config.decoder_input_dim, 1),
        )?;
        let speaker_encoder = ResNetSpeakerEncoder::new(
            vb.pp("speaker_encoder"),
            &ResNetSpeakerEncoderConfig::default(),
        )?;

        Ok(Self {
            config: config.clone(),
            waveform_decoder,
            speaker_encoder,
        })
    }

    pub fn forward(&self, latents: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut z = latents.transpose(1, 2)?;
        z =
            z.interpolate1d(self.config.ar_mel_length_compression / self.config.output_hop_length)?;
        z = z.squeeze(1)?;

        if self.config.output_sample_rate != self.config.input_sample_rate {
            z = z.interpolate1d(self.config.output_sample_rate / self.config.input_sample_rate)?;
        }

        let o = self.waveform_decoder.forward(&z, g)?;
        Ok(o)
    }
}
