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
