use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::init::Init;
use candle_nn::var_builder::SimpleBackend;

pub struct Ones;

impl SimpleBackend for Ones {
    fn get(&self, s: Shape, _: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        Tensor::ones(s, dtype, dev)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

pub fn flatten_4d<T: Clone>(vec_4d: &[Vec<Vec<Vec<T>>>]) -> Vec<T> {
    vec_4d
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .cloned()
        .collect()
}

pub fn flatten_3d<T: Clone>(vec_3d: &[Vec<Vec<T>>]) -> Vec<T> {
    vec_3d.iter().flatten().flatten().cloned().collect()
}

pub fn flatten_2d<T: Clone>(vec_2d: &[Vec<T>]) -> Vec<T> {
    vec_2d.iter().flatten().cloned().collect()
}
