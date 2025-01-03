use candle_core::{Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{self as nn, linear, seq, BatchNormConfig, Conv1d, Conv2d, Linear, Sequential};
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

    fn load_se_layer_fc(vb: VarBuilder, i: usize, o: usize) -> Result<Self> {
        let ws = vb.get((o, i), "weight")?;
        let ws = ws.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let bs = vb.get(o, "bias")?;
        let config = nn::Conv2dConfig {
            stride: 1,
            groups: 1,
            ..Default::default()
        };
        let conv2d = Conv2d::new(ws, Some(bs), config);
        Ok(Self { conv2d, s: 1, k: 1 })
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

#[derive(Debug, Clone)]
pub struct DownsampleBlock {
    conv: Conv2d,
    bn: nn::BatchNorm,
}

impl DownsampleBlock {
    pub fn new(vb: VarBuilder, i: usize, o: usize, k: usize, stride: usize) -> Result<Self> {
        let conv = nn::conv2d_no_bias(
            i,
            o,
            k,
            nn::Conv2dConfig {
                stride,
                ..Default::default()
            },
            vb.pp("0"),
        )?;
        let bn = nn::batch_norm(o, BatchNormConfig::default(), vb.pp("1"))?;
        Ok(Self { conv, bn })
    }
}

impl ModuleT for DownsampleBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.bn.forward_t(&xs, train)?;
        Ok(xs)
    }
}

pub struct SELayerConfig {
    channels: usize,
    reduction: usize,
}

impl SELayerConfig {
    pub fn new(channels: usize, reduction: usize) -> Self {
        Self {
            channels,
            reduction,
        }
    }

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
        let fc1 = Conv2DSame::load_se_layer_fc(
            vb.pp("fc.0"),
            config.channels,
            config.channels / config.reduction,
        )?;
        let fc2 = Conv2DSame::load_se_layer_fc(
            vb.pp("fc.2"),
            config.channels / config.reduction,
            config.channels,
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

pub struct SEBasicBlockConfig {
    inplanes: usize,
    planes: usize,
    stride: usize,
    downsample: Option<DownsampleBlock>,
    reduction: usize,
}

impl SEBasicBlockConfig {
    pub fn default(inplanes: usize, planes: usize, stride: usize) -> Self {
        Self {
            inplanes,
            planes,
            stride,
            downsample: None,
            reduction: 8,
        }
    }
}

pub struct SEBasicBlock {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    conv2: Conv2d,
    bn2: nn::BatchNorm,
    se: SELayer,
    downsample: Option<DownsampleBlock>,
}

impl SEBasicBlock {
    pub fn new(vb: VarBuilder, config: &SEBasicBlockConfig) -> Result<Self> {
        let conv1 = nn::conv2d_no_bias(
            config.inplanes,
            config.planes,
            3,
            nn::Conv2dConfig {
                stride: config.stride,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let bn1 = nn::batch_norm(config.planes, BatchNormConfig::default(), vb.pp("bn1"))?;

        let conv2 = nn::conv2d_no_bias(
            config.planes,
            config.planes,
            3,
            nn::Conv2dConfig {
                stride: config.stride,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        let bn2 = nn::batch_norm(config.planes, BatchNormConfig::default(), vb.pp("bn2"))?;
        let se = SELayer::new(
            vb.pp("se"),
            &SELayerConfig::new(config.planes, config.reduction),
        )?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            se,
            downsample: config.downsample.clone(),
        })
    }
}

impl Module for SEBasicBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.conv1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.bn1.forward_t(&xs, false)?;

        let xs = self.conv2.forward(&xs)?;
        let xs = self.bn2.forward_t(&xs, false)?;
        let xs = self.se.forward(&xs)?;

        let residual = if let Some(downsample) = &self.downsample {
            &downsample.forward_t(residual, false)?
        } else {
            residual
        };

        let xs = (xs + residual)?;
        let xs = xs.relu()?;
        Ok(xs)
    }
}

pub struct ResNetAttentionBlock {
    conv1: Conv1d,
    bn1: nn::BatchNorm,
    conv2: Conv1d,
}

impl ResNetAttentionBlock {
    pub fn new(vb: VarBuilder, i: usize, o: usize) -> Result<Self> {
        let conv1 = nn::conv1d(
            i,
            128,
            1,
            nn::Conv1dConfig {
                ..Default::default()
            },
            vb.pp("0"),
        )?;
        let bn1 = nn::batch_norm(128, BatchNormConfig::default(), vb.pp("2"))?;
        let conv2 = nn::conv1d_no_bias(
            128,
            o,
            1,
            nn::Conv1dConfig {
                ..Default::default()
            },
            vb.pp("3"),
        )?;
        Ok(Self { conv1, bn1, conv2 })
    }
}

impl ModuleT for ResNetAttentionBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.bn1.forward_t(&xs, train)?;
        let xs = self.conv2.forward(&xs)?;
        let xs = softmax(&xs, 2)?;
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
    l2_norm: bool,
}

impl Default for ResNetSpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            proj_dim: 512,
            layers: vec![3, 4, 6, 3],
            num_filters: vec![32, 64, 128, 256],
            encoder_type: "ASP".to_string(),
            log_input: true,
            use_torch_spec: false,
            l2_norm: true,
        }
    }
}

fn create_layer(
    vb: VarBuilder,
    inplanes: usize,
    planes: usize,
    num_blocks: usize,
    stride: usize,
) -> Sequential {
    let downsample = if stride != 1 || inplanes != planes {
        Some(DownsampleBlock::new(vb.pp("0.downsample"), inplanes, planes, 1, stride).unwrap())
    } else {
        None
    };

    let mut layers = seq();
    layers = layers.add(
        SEBasicBlock::new(
            vb.pp("0"),
            &SEBasicBlockConfig {
                inplanes,
                planes,
                stride,
                downsample,
                reduction: 8,
            },
        )
        .unwrap(),
    );

    let inplanes = planes;
    for i in 1..num_blocks {
        layers = layers.add(
            SEBasicBlock::new(
                vb.pp(i.to_string()),
                &SEBasicBlockConfig {
                    inplanes,
                    planes,
                    stride: 1,
                    downsample: None,
                    reduction: 8,
                },
            )
            .unwrap(),
        );
    }

    layers
}

pub struct ResNetSpeakerEncoder {
    conv1: Conv2d,
    bn1: nn::BatchNorm,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    attention: ResNetAttentionBlock,
    fc: Linear,
}

impl ResNetSpeakerEncoder {
    pub fn new(vb: VarBuilder, config: &ResNetSpeakerEncoderConfig) -> Result<Self> {
        let conv1 = nn::conv2d(
            1,
            config.num_filters[0],
            3,
            nn::Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let bn1 = nn::batch_norm(
            config.num_filters[0],
            BatchNormConfig::default(),
            vb.pp("bn1"),
        )?;

        let layer1 = create_layer(
            vb.pp("layer1"),
            config.num_filters[0],
            config.num_filters[0],
            config.layers[0],
            1,
        );
        let layer2 = create_layer(
            vb.pp("layer2"),
            config.num_filters[0],
            config.num_filters[1],
            config.layers[1],
            2,
        );
        let layer3 = create_layer(
            vb.pp("layer3"),
            config.num_filters[1],
            config.num_filters[2],
            config.layers[2],
            2,
        );
        let layer4 = create_layer(
            vb.pp("layer4"),
            config.num_filters[2],
            config.num_filters[3],
            config.layers[3],
            2,
        );

        let outmap_size = config.input_dim / 8;
        let attention = ResNetAttentionBlock::new(
            vb.pp("attention"),
            config.num_filters[3] * outmap_size,
            config.num_filters[3] * outmap_size,
        )?;

        let fc = linear(
            config.num_filters[3] * outmap_size * 2, // self.encoder_type == "ASP":
            config.proj_dim,
            vb.pp("fc"),
        )?;

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            attention,
            fc,
        })
    }
}

impl Module for ResNetSpeakerEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.bn1.forward_t(&xs, false)?;

        let xs = self.layer1.forward(&xs)?;
        let xs = self.layer2.forward(&xs)?;
        let xs = self.layer3.forward(&xs)?;
        let xs = self.layer4.forward(&xs)?;

        let new_shape = (xs.shape().dim(0)?, (), xs.shape().dim(D::Minus1)?);
        let xs = xs.reshape(new_shape)?;

        let xs = self.attention.forward_t(&xs, false)?;

        let xs = self.fc.forward(&xs)?;

        Ok(xs)
    }
}
