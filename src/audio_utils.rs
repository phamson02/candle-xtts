use anyhow::Result;
use candle_core::{Device, Tensor};
use libsoxr::Soxr;
use wavers::Wav;

fn resample_vec(
    audio: &[f32],
    n_channels: u32,
    orig_freq: u32,
    target_freq: u32,
) -> Result<Vec<f32>> {
    let soxr = Soxr::create(
        orig_freq as f64,
        target_freq as f64,
        n_channels,
        None,
        None,
        None,
    )
    .unwrap();
    let mut target =
        vec![0.0; (audio.len() as f64 * target_freq as f64 / orig_freq as f64) as usize];
    soxr.process(Some(audio), &mut target).unwrap();
    soxr.process::<f32, _>(None, &mut target[0..]).unwrap();
    Ok(target)
}

pub fn resample_tensor(
    audio: &Tensor,
    n_channels: u32,
    orig_freq: u32,
    target_freq: u32,
) -> Result<Tensor> {
    let audio = audio.to_vec1::<f32>()?;
    let resampled = resample_vec(&audio, n_channels, orig_freq, target_freq)?;
    let resampled_len = resampled.len();
    let resampled_tensor = Tensor::from_vec(resampled, (1, resampled_len), &Device::Cpu)?;
    Ok(resampled_tensor)
}

pub fn load_audio(audio_path: &str, sampling_rate: u32) -> Result<Tensor> {
    let mut wav: Wav<f32> = Wav::from_path(audio_path)?;
    let lsr = wav.sample_rate() as u32;
    let is_mono = wav.n_channels() == 1;
    let samples: &[f32] = &wav.read()?;

    // Stereo to mono if necessary
    let audio: Vec<f32> = if is_mono {
        samples.to_vec()
    } else {
        samples
            .chunks(2)
            .map(|stereo| (stereo[0] + stereo[1]) / 2.0)
            .collect()
    };

    // Resample if necessary
    let audio = if lsr == sampling_rate {
        audio
    } else {
        resample_vec(&audio, 1, lsr, sampling_rate)?
    };

    if audio.iter().any(|&x| x > 10.0) || !audio.iter().any(|&x| x < 0.0) {
        panic!(
            "Error with {}. Max={}, Min={}",
            audio_path,
            audio
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            audio
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        );
    }

    let audio_len = audio.len();
    let audio_tensor = Tensor::from_vec(audio, (1, audio_len), &Device::Cpu)?;

    // clip audio tensor to [-1, 1]
    let audio_tensor = audio_tensor.clamp(-1.0, 1.0)?;
    Ok(audio_tensor)
}
