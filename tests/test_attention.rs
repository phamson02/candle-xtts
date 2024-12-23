use serde::Deserialize;

#[derive(Deserialize)]
struct TestData {
    x: Vec<Vec<Vec<f64>>>,
    context: Vec<Vec<Vec<f64>>>,
    mask: Vec<Vec<bool>>,
    output: Vec<Vec<Vec<f64>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_xtts::layers::{
        perceiver_encoder::{Attention, AttentionConfig},
        utils::{flatten_2d, flatten_3d, Ones},
    };
    use serde_json;

    #[test]
    fn test_attention() {
        let json_data = include_str!("./test_samples/attention.json");

        let test_data: TestData =
            serde_json::from_str(json_data).expect("JSON was not well-formatted");

        let x = Tensor::from_vec(
            flatten_3d(&test_data.x),
            (
                test_data.x.len(),
                test_data.x[0].len(),
                test_data.x[0][0].len(),
            ),
            &Device::Cpu,
        )
        .unwrap();

        let context = Tensor::from_vec(
            flatten_3d(&test_data.context),
            (
                test_data.context.len(),
                test_data.context[0].len(),
                test_data.context[0][0].len(),
            ),
            &Device::Cpu,
        )
        .unwrap();

        let mask = Tensor::from_vec(
            flatten_2d(&test_data.mask)
                .iter()
                .map(|&x| if !x { f64::NEG_INFINITY } else { 0f64 })
                .collect(),
            (test_data.mask.len(), test_data.mask[0].len()),
            &Device::Cpu,
        )
        .unwrap();

        let expected_output = flatten_3d(&test_data.output);

        let vb = VarBuilder::from_backend(Box::new(Ones), DType::F64, Device::Cpu);

        let module = Attention::new(vb, &AttentionConfig::default()).unwrap();
        let output = module.forward(&x, &context, Some(&mask)).unwrap();
        let output = output.flatten_all().unwrap().to_vec1().unwrap();

        dbg!(&output);

        assert_tensors_approx_equal(&output, &expected_output, Some(1e-1));
    }

    fn assert_tensors_approx_equal(output: &Vec<f64>, expected: &Vec<f64>, tol: Option<f64>) {
        assert_eq!(output.len(), expected.len(), "Size mismatch");
        let tol = tol.unwrap_or(1e-6);
        for (i, (out_val, exp_val)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (out_val - exp_val).abs();
            assert!(
                diff <= tol,
                "Value mismatch at channel {}, index {}: {} vs {}",
                0,
                i,
                out_val,
                exp_val
            );
        }
    }
}
