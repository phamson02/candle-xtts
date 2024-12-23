use serde::Deserialize;

#[derive(Deserialize)]
struct TestData {
    input: Vec<Vec<Vec<f64>>>,
    output: Vec<Vec<Vec<f64>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_xtts::layers::hifigan_generator::*;
    use candle_xtts::layers::utils::Ones;
    use serde_json;

    #[test]
    fn test_res_block1_forward() {
        let json_data = include_str!("./test_samples/hifigan_generator_res1.json");

        let test_data: TestData =
            serde_json::from_str(json_data).expect("JSON was not well-formatted");

        let input = &test_data.input[0];
        let expected_output = &test_data.output[0];

        let res1_config = ResBlock1Config::default(2);
        let res1_vb = VarBuilder::from_backend(Box::new(Ones), DType::F64, Device::Cpu);
        let res1 = ResBlock1::new(res1_vb, &res1_config).unwrap();

        let input_tensor = Tensor::from_vec(
            input.iter().flatten().copied().collect(),
            (1, input.len(), input[0].len()),
            &Device::Cpu,
        )
        .unwrap();

        let output_tensor = res1.forward(&input_tensor).unwrap();
        let output = output_tensor.to_vec3().unwrap();

        dbg!(&output);

        assert_tensors_approx_equal(&output[0], &expected_output, Some(1e-4));
    }

    #[test]
    fn test_res_block2_forward() {
        let json_data = include_str!("./test_samples/hifigan_generator_res2.json");

        let test_data: TestData =
            serde_json::from_str(json_data).expect("JSON was not well-formatted");

        let input = &test_data.input[0];
        let expected_output = &test_data.output[0];

        let res2_config = ResBlock2Config::default(2);
        let res2_vb = VarBuilder::from_backend(Box::new(Ones), DType::F64, Device::Cpu);
        let res2 = ResBlock2::new(res2_vb, &res2_config).unwrap();

        let input_tensor = Tensor::from_vec(
            input.iter().flatten().copied().collect(),
            (1, input.len(), input[0].len()),
            &Device::Cpu,
        )
        .unwrap();

        let output_tensor = res2.forward(&input_tensor).unwrap();
        let output = output_tensor.to_vec3().unwrap();

        dbg!(&output);

        assert_tensors_approx_equal(&output[0], &expected_output, None);
    }

    fn assert_tensors_approx_equal(
        output: &Vec<Vec<f64>>,
        expected: &Vec<Vec<f64>>,
        tol: Option<f64>,
    ) {
        assert_eq!(output.len(), expected.len(), "Batch size mismatch");
        let tol = tol.unwrap_or(1e-6);
        for (batch_idx, (out_channel, exp_channel)) in
            output.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(
                out_channel.len(),
                exp_channel.len(),
                "Channel count mismatch at batch {}",
                batch_idx
            );
            for (i, (out_val, exp_val)) in out_channel.iter().zip(exp_channel.iter()).enumerate() {
                let diff = (out_val - exp_val).abs();
                assert!(
                    diff <= tol,
                    "Value mismatch at batch {}, channel {}, index {}: {} vs {}",
                    batch_idx,
                    0,
                    i,
                    out_val,
                    exp_val
                );
            }
        }
    }
}
