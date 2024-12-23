use candle_core::{Result, Tensor, D};
use candle_nn as nn;
use candle_nn::{ops::sigmoid, Module, ModuleT, VarBuilder};

#[derive(Debug)]
struct Conv2DSame {
    conv2d: nn::Conv2d,
    s: usize,
    k: usize,
}

impl Conv2DSame {
    fn new(
        vb: VarBuilder,
        i: usize,
        o: usize,
        k: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv_config = nn::Conv2dConfig {
            stride,
            groups,
            ..Default::default()
        };
        let conv2d = if bias {
            nn::conv2d(i, o, k, conv_config, vb)?
        } else {
            nn::conv2d_no_bias(i, o, k, conv_config, vb)?
        };
        Ok(Self {
            conv2d,
            s: stride,
            k,
        })
    }
}

impl Module for Conv2DSame {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let s = self.s;
        let k = self.k;
        let (_, _, ih, iw) = xs.dims4()?;
        let oh = (ih + s - 1) / s;
        let ow = (iw + s - 1) / s;
        let pad_h = usize::max((oh - 1) * s + k - ih, 0);
        let pad_w = usize::max((ow - 1) * s + k - iw, 0);
        if pad_h > 0 || pad_w > 0 {
            let xs = xs.pad_with_zeros(2, pad_h / 2, pad_h - pad_h / 2)?;
            let xs = xs.pad_with_zeros(3, pad_w / 2, pad_w - pad_w / 2)?;
            self.conv2d.forward(&xs)
        } else {
            self.conv2d.forward(xs)
        }
    }
}

pub struct SELayerConfig {
    channels: usize,
    reduction: usize,
}

impl SELayerConfig {
    pub fn default(channels: usize) -> Self {
        Self {
            channels,
            reduction: 8,
        }
    }
}

pub struct SELayer {
    fc1: Conv2DSame,
    fc2: Conv2DSame,
}

impl SELayer {
    pub fn new(vb: VarBuilder, config: &SELayerConfig) -> Result<Self> {
        let fc1 = Conv2DSame::new(
            vb.pp("fc1"),
            config.channels,
            config.channels / config.reduction,
            1,
            1,
            1,
            true,
        )?;
        let fc2 = Conv2DSame::new(
            vb.pp("fc2"),
            config.channels / config.reduction,
            config.channels,
            1,
            1,
            1,
            true,
        )?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for SELayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        // equivalent to adaptive_avg_pool2d([1, 1])
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = self.fc2.forward(&xs)?;
        let xs = sigmoid(&xs)?;
        residual.broadcast_mul(&xs)
    }
}

struct SEBasicBlockConfig {
    inplanes: usize,
    planes: usize,
    stride: usize,
    downsample: Option<Conv2DSame>,
    reduction: usize,
}

impl SEBasicBlockConfig {
    pub fn default(inplanes: usize, planes: usize) -> Self {
        Self {
            inplanes,
            planes,
            stride: 1,
            downsample: None,
            reduction: 8,
        }
    }
}

pub struct SEBasicBlock {
    conv1: Conv2DSame,
    bn1: nn::BatchNorm,
    conv2: Conv2DSame,
    bn2: nn::BatchNorm,
    se: SELayer,
    downsample: Option<Conv2DSame>,
}

impl SEBasicBlock {
    pub fn new(vb: VarBuilder, config: &SEBasicBlock) -> Result<Self> {
        todo!()
    }
}

impl ModuleT for SEBasicBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let residual = xs;
        let xs = self.conv1.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = self.bn1.forward_t(&xs, train)?;

        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, train)?;
        let xs = self.se.forward(&xs)?;

        let residual = if let Some(downsample) = &self.downsample {
            &downsample.forward(&residual)?
        } else {
            residual
        };

        let xs = (xs + residual)?;
        let xs = xs.relu()?;
        Ok(xs)
    }
}
pub struct ResNetSpeakerEncoderConfig {
    input_dim: usize,
    proj_dim: usize,
    layers: Vec<usize>,
    num_filters: Vec<usize>,
    encoder_type: String,
    log_input: bool,
    use_torch_spec: bool,
}
