use serde::Deserialize;

#[derive(Deserialize)]
struct TestData {
    input: Vec<Vec<Vec<f64>>>,
    output: Vec<Vec<Vec<f64>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor, D};
    use candle_nn::VarBuilder;
    use candle_xtts::layers::{
        perceiver_encoder::RMSNorm,
        utils::{flatten_3d, Ones},
    };
    use serde_json;

    #[test]
    fn test_rms_norm() {
        let json_data = include_str!("./test_samples/rms_norm.json");

        let test_data: TestData =
            serde_json::from_str(json_data).expect("JSON was not well-formatted");

        let x = Tensor::from_vec(
            flatten_3d(&test_data.input),
            (
                test_data.input.len(),
                test_data.input[0].len(),
                test_data.input[0][0].len(),
            ),
            &Device::Cpu,
        )
        .unwrap();

        let expected_output = flatten_3d(&test_data.output);

        let vb = VarBuilder::from_backend(Box::new(Ones), DType::F64, Device::Cpu);
        let module = RMSNorm::new(vb, x.shape().dim(D::Minus1).unwrap()).unwrap();
        let output = module.forward(&x).unwrap();
        let output = output.flatten_all().unwrap().to_vec1().unwrap();

        dbg!(&output);

        assert_tensors_approx_equal(&output, &expected_output, None);
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
