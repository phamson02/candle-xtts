use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::init::Init;
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Conv1d, VarBuilder};

pub struct Ones;

impl SimpleBackend for Ones {
    fn get(&self, s: Shape, _: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        Tensor::ones(s, dtype, dev)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

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

pub fn flatten_4d<T: Clone>(vec_4d: &Vec<Vec<Vec<Vec<T>>>>) -> Vec<T> {
    vec_4d
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .cloned()
        .collect()
}

pub fn flatten_3d<T: Clone>(vec_3d: &Vec<Vec<Vec<T>>>) -> Vec<T> {
    vec_3d.iter().flatten().flatten().cloned().collect()
}

pub fn flatten_2d<T: Clone>(vec_2d: &Vec<Vec<T>>) -> Vec<T> {
    vec_2d.iter().flatten().cloned().collect()
}
