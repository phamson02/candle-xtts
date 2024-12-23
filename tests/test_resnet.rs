use serde::Deserialize;

#[derive(Deserialize)]
struct TestData {
    input: Vec<Vec<Vec<Vec<f64>>>>,
    output: Vec<Vec<Vec<Vec<f64>>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Module, Tensor};
    use candle_nn::VarBuilder;
    use candle_xtts::layers::resnet::*;
    use candle_xtts::layers::utils::Ones;
    use serde_json;

    #[test]
    fn test_se_layer_forward() {
        let json_data = include_str!("./test_samples/se_layer.json");

        let test_data: TestData =
            serde_json::from_str(json_data).expect("JSON was not well-formatted");

        let input = &test_data.input[0];
        let expected_output = &test_data
            .output
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        let config = SELayerConfig::default(64);
        let vb = VarBuilder::from_backend(Box::new(Ones), DType::F64, Device::Cpu);
        let block = SELayer::new(vb, &config).unwrap();

        let input_tensor = Tensor::from_vec(
            input.iter().flatten().flatten().copied().collect(),
            (1, input.len(), input[0].len(), input[0][0].len()),
            &Device::Cpu,
        )
        .unwrap();

        let output_tensor = block.forward(&input_tensor).unwrap();
        let output: Vec<f64> = output_tensor.flatten_all().unwrap().to_vec1().unwrap();

        dbg!(&output);

        assert_tensors_approx_equal(&output, expected_output, None);
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
