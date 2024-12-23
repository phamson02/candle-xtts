use serde::Deserialize;

#[derive(Deserialize)]
struct TestData {
    q: Vec<Vec<Vec<Vec<f64>>>>, // [batch_size, num_heads, query_length, feature_dim]
    k: Vec<Vec<Vec<Vec<f64>>>>, // [batch_size, num_heads, key_length, feature_dim]
    v: Vec<Vec<Vec<Vec<f64>>>>, // [batch_size, num_heads, key_length, feature_dim]
    mask: Vec<Vec<bool>>,       // [batch_size, key_length]
    output: Vec<Vec<Vec<Vec<f64>>>>, // [batch_size, num_heads, query_length, feature_dim]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_xtts::layers::{
        perceiver_encoder::Attend,
        utils::{flatten_2d, flatten_4d},
    };
    use serde_json;

    #[test]
    fn test_attend() {
        let json_data = include_str!("./test_samples/attend.json");

        let test_data: TestData =
            serde_json::from_str(json_data).expect("JSON was not well-formatted");

        let q = Tensor::from_vec(
            flatten_4d(&test_data.q),
            (
                test_data.q.len(),
                test_data.q[0].len(),
                test_data.q[0][0].len(),
                test_data.q[0][0][0].len(),
            ),
            &Device::Cpu,
        )
        .unwrap();

        let k = Tensor::from_vec(
            flatten_4d(&test_data.k),
            (
                test_data.k.len(),
                test_data.k[0].len(),
                test_data.k[0][0].len(),
                test_data.k[0][0][0].len(),
            ),
            &Device::Cpu,
        )
        .unwrap();

        let v = Tensor::from_vec(
            flatten_4d(&test_data.v),
            (
                test_data.v.len(),
                test_data.v[0].len(),
                test_data.v[0][0].len(),
                test_data.v[0][0][0].len(),
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

        let expected_output = flatten_4d(&test_data.output);

        let module = Attend {};
        let output = module.forward(&q, &k, &v, Some(&mask)).unwrap();
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
