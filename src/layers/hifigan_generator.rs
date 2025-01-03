use candle_core::{Result, Tensor};
use candle_nn::ops::leaky_relu;
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use std::iter::Iterator;

pub fn conv1d(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_c, in_c, kernel_size), "weight")?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(
        (out_c, in_c, kernel_size),
        "parametrizations.weight.original1",
    )?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight_g = vb.get((in_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(
        (in_c, out_c, kernel_size),
        "parametrizations.weight.original1",
    )?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, config))
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

pub struct ResBlock1Config {
    channels: usize,
    kernel_size: usize,
    dilation: [usize; 3],
}

impl ResBlock1Config {
    pub fn default(channels: usize) -> Self {
        Self {
            channels,
            kernel_size: 3,
            dilation: [1, 3, 5],
        }
    }
}

pub struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    pub fn new(vb: VarBuilder, config: &ResBlock1Config) -> Result<Self> {
        let convs1 = config
            .dilation
            .iter()
            .enumerate()
            .map(|(i, &dilation)| {
                conv1d_weight_norm(
                    config.channels,
                    config.channels,
                    config.kernel_size,
                    true,
                    Conv1dConfig {
                        stride: 1,
                        padding: get_padding(config.kernel_size, dilation),
                        dilation,
                        groups: 1,
                    },
                    vb.pp(format!("convs1.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let convs2 = (0..convs1.len())
            .map(|i| {
                conv1d_weight_norm(
                    config.channels,
                    config.channels,
                    config.kernel_size,
                    true,
                    Conv1dConfig {
                        stride: 1,
                        padding: get_padding(config.kernel_size, 1),
                        dilation: 1,
                        groups: 1,
                    },
                    vb.pp(format!("convs2.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { convs1, convs2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        std::iter::zip(self.convs1.iter(), self.convs2.iter()).try_fold(
            x.clone(),
            |acc, (conv1, conv2)| -> Result<Tensor> {
                let mut xt = leaky_relu(&acc, 0.1)?;
                xt = conv1.forward(&xt)?;
                xt = leaky_relu(&xt, 0.1)?;
                xt = conv2.forward(&xt)?;
                xt + acc
            },
        )
    }
}

pub struct ResBlock2Config {
    channels: usize,
    kernel_size: usize,
    dilation: [usize; 2],
}

impl ResBlock2Config {
    pub fn default(channels: usize) -> Self {
        Self {
            channels,
            kernel_size: 3,
            dilation: [1, 3],
        }
    }
}

pub struct ResBlock2 {
    convs: Vec<Conv1d>,
}

impl ResBlock2 {
    pub fn new(vb: VarBuilder, config: &ResBlock2Config) -> Result<Self> {
        let convs = config
            .dilation
            .iter()
            .enumerate()
            .map(|(i, &dilation)| {
                conv1d_weight_norm(
                    config.channels,
                    config.channels,
                    config.kernel_size,
                    true,
                    Conv1dConfig {
                        stride: 1,
                        padding: get_padding(config.kernel_size, dilation),
                        dilation,
                        groups: 1,
                    },
                    vb.pp(format!("conv.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { convs })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.convs.iter().try_fold(x.clone(), |acc, conv| {
            let mut xt = leaky_relu(&acc, 0.1)?;
            xt = conv.forward(&xt)?;
            xt + acc
        })
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ResBlockType {
    ResBlock1,
    ResBlock2,
}

pub enum ResBlock {
    ResBlock1(ResBlock1),
    ResBlock2(ResBlock2),
}

impl ResBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            ResBlock::ResBlock1(resblock) => resblock.forward(x),
            ResBlock::ResBlock2(resblock) => resblock.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HifiganGeneratorConfig {
    in_channels: usize,
    out_channels: usize,
    resblock_type: ResBlockType,
    resblock_dilation_sizes: Vec<Vec<usize>>,
    resblock_kernel_sizes: Vec<usize>,
    upsample_kernel_sizes: Vec<usize>,
    upsample_initial_channel: usize,
    upsample_factors: Vec<usize>,
    inference_padding: usize,
    cond_channels: usize,
    conv_post_bias: bool,
    cond_in_each_up_layer: bool,
}

impl HifiganGeneratorConfig {
    pub fn default(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            resblock_type: ResBlockType::ResBlock1,
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            resblock_kernel_sizes: vec![3, 7, 11],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            upsample_initial_channel: 512,
            upsample_factors: vec![8, 8, 2, 2],
            inference_padding: 0,
            cond_channels: 512,
            conv_post_bias: false,
            cond_in_each_up_layer: true,
        }
    }
}

pub struct HifiganGenerator {
    num_kernels: usize,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    cond_layer: Option<Conv1d>,
    conds: Option<Vec<Conv1d>>,
}

impl HifiganGenerator {
    pub fn new(vb: VarBuilder, config: &HifiganGeneratorConfig) -> Result<Self> {
        let inference_padding = config.inference_padding;
        let num_kernels = config.upsample_kernel_sizes.len();
        let num_upsamples = config.upsample_factors.len();
        let cond_in_each_up_layer = config.cond_in_each_up_layer;
        let conv_pre = conv1d(
            config.in_channels,
            config.upsample_initial_channel,
            7,
            true,
            Conv1dConfig {
                stride: 1,
                padding: 3,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv_pre"),
        )?;

        let ups = std::iter::zip(
            config.upsample_factors.iter(),
            config.upsample_kernel_sizes.iter(),
        )
        .enumerate()
        .map(|(i, (&u, &k))| {
            conv_transpose1d_weight_norm(
                config.upsample_initial_channel / 2_usize.pow(i as u32),
                config.upsample_initial_channel / 2_usize.pow(i as u32 + 1),
                k,
                true,
                ConvTranspose1dConfig {
                    stride: u,
                    padding: (k - u) / 2,
                    output_padding: 0,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("ups.{i}")),
            )
        })
        .collect::<Result<Vec<_>>>()?;

        let resblocks = (0..num_upsamples)
            .flat_map(|i| {
                let ch = config.upsample_initial_channel / 2_usize.pow(i as u32 + 1);
                std::iter::zip(
                    config.resblock_kernel_sizes.iter(),
                    config.resblock_dilation_sizes.iter(),
                )
                .enumerate()
                .map(|(j, (&k, d))| match config.resblock_type {
                    ResBlockType::ResBlock1 => {
                        let resblock_config = ResBlock1Config {
                            channels: ch,
                            kernel_size: k,
                            dilation: [d[0], d[1], d[2]],
                        };
                        ResBlock::ResBlock1(
                            ResBlock1::new(
                                vb.pp(format!(
                                    "resblocks.{}",
                                    i * config.resblock_kernel_sizes.len() + j
                                )),
                                &resblock_config,
                            )
                            .unwrap(),
                        )
                    }
                    ResBlockType::ResBlock2 => {
                        let resblock_config = ResBlock2Config {
                            channels: ch,
                            kernel_size: k,
                            dilation: [d[0], d[1]],
                        };
                        ResBlock::ResBlock2(
                            ResBlock2::new(
                                vb.pp(format!(
                                    "resblocks.{}",
                                    i * config.resblock_kernel_sizes.len() + j
                                )),
                                &resblock_config,
                            )
                            .unwrap(),
                        )
                    }
                })
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let conv_post = conv1d(
            config.upsample_initial_channel / 2_usize.pow(num_upsamples as u32),
            config.out_channels,
            7,
            config.conv_post_bias,
            Conv1dConfig {
                stride: 1,
                padding: 3,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv_post"),
        )?;

        let cond_layer = if config.cond_channels > 0 {
            Some(conv1d(
                config.cond_channels,
                config.upsample_initial_channel,
                1,
                true,
                Conv1dConfig::default(),
                vb.pp("cond_layer"),
            )?)
        } else {
            None
        };

        let conds = if cond_in_each_up_layer {
            let conds = (0..num_upsamples)
                .map(|i| {
                    conv1d(
                        config.cond_channels,
                        config.upsample_initial_channel / 2_usize.pow(i as u32 + 1),
                        1,
                        true,
                        Conv1dConfig::default(),
                        vb.pp(format!("conds.{i}")),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Some(conds)
        } else {
            None
        };

        Ok(Self {
            num_kernels,
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond_layer,
            conds,
        })
    }

    pub fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut o = self.conv_pre.forward(x)?;

        if let Some(cond_layer) = self.cond_layer.as_ref() {
            o = (o + cond_layer.forward(g.unwrap())?)?;
        }

        for i in 0..self.ups.len() {
            o = leaky_relu(&o, 0.1)?;
            o = self.ups[i].forward(&o)?;
            o = if let Some(conds) = self.conds.as_ref() {
                (o + conds[i].forward(g.unwrap())?)?
            } else {
                o
            };
            let z_sum_init = self.resblocks[i * self.num_kernels].forward(&o)?;
            let z_sum = (1..self.num_kernels).try_fold(z_sum_init, |z_sum, j| {
                let z = self.resblocks[i * self.num_kernels + j].forward(&o)?;
                z_sum + z
            })?;
            o = (z_sum / self.num_kernels as f64)?;
            o = leaky_relu(&o, 0.1)?;
            o = self.conv_post.forward(&o)?;
            o = o.tanh()?;
        }

        Ok(o)
    }

    pub fn inference(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
