use std::fmt::Debug;

use ndarray::{array, Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64>;
}

#[derive(Debug)]
pub struct FCLayer {
    weights: Array2<f64>,
    bias: Array2<f64>,
    x: Array2<f64>,
    d_weights: Array2<f64>,
    d_bias: Array1<f64>,
}

impl FCLayer {
    pub fn new(input_size: usize, output_size: usize) -> Box<FCLayer> {
        let fc = FCLayer {
            weights: Array::random((input_size, output_size), StandardNormal),
            bias: Array::random((1, output_size), StandardNormal),
            x: array![[0.0]],
            d_weights: array![[0.0]],
            d_bias: array![0.0],
        };
        Box::new(fc)
    }
}

impl Layer for FCLayer {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.x = input.clone();
        input.dot(&self.weights) + &self.bias
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        let dx = dout.dot(&self.weights.t());
        self.d_weights = self.x.t().dot(dout);
        // self.d_bias = dout.sum_axis(Axis(0)).into_shape((1, dout.ncols())).unwrap();
        self.d_bias = dout.sum_axis(Axis(0));

        dx
    }
}

#[derive(Debug)]
pub struct Relu {}

impl Relu {
    pub fn new() -> Box<Relu> {
        Box::new(Relu {})
    }
}

impl Layer for Relu {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        dout.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Debug)]
pub struct Sigmoid {
    out: Array2<f64>,
}

impl Sigmoid {
    pub fn new() -> Box<Sigmoid> {
        Box::new(Sigmoid { out: array![[0.0]] })
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let out = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.out = out.clone();

        out
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        let temp = 1.0 - &self.out;
        println!("{:?}", temp);
        let dx = dout * (1.0 - &self.out) * &self.out;

        dx
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis};

    use super::{FCLayer, Layer, Relu, Sigmoid};
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_fc_forward() {
        let expected = array![[36.0], [52.0]];

        let mut fc = FCLayer::new(2, 2);
        fc.weights = array![[1.0, 2.0], [3.0, 4.0]];
        fc.bias = array![[5.0, 6.0]];
        let input = array![[7.0, 8.0]];

        let actual = fc.forward(&input);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e, 1e-3f64);
        }
    }

    #[test]
    fn test_fc_backward() {
        let dx_expected = array![[11.0, 25.0], [17.0, 39.0]];
        let dw_expected = array![[18.0, 22.0], [26.0, 32.0]];
        let db_expected = array![8.0, 10.0];

        let mut fc = FCLayer::new(2, 2);
        fc.weights = array![[1.0, 2.0], [3.0, 4.0]];
        fc.bias = array![[5.0, 6.0]];
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        fc.x = input.clone();
        let dout = array![[3.0, 4.0], [5.0, 6.0]];
        let dx_actual = fc.backward(&dout);

        for (a, e) in dx_actual.iter().zip(dx_expected.iter()) {
            assert_approx_eq!(a, e, 1e-3f64);
        }

        for (a, e) in fc.d_weights.iter().zip(dw_expected.iter()) {
            assert_approx_eq!(a, e, 1e-3f64);
        }

        for (a, e) in fc.d_bias.iter().zip(db_expected.iter()) {
            assert_approx_eq!(a, e, 1e-3f64);
        }
    }

    #[test]
    fn test_relu_forward() {
        let expected = array![[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]];

        let mut relu = Relu::new();
        let input = array![[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]];

        let actual = relu.forward(&input);

        assert_eq!(actual, expected);
    }
    #[test]
    fn test_relu_backward() {
        let expected = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];

        let mut relu = Relu::new();
        let input = array![[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]];

        let actual = relu.backward(&input);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_sigmoid_forward() {
        let expected = array![[0.269, 0.5], [0.731, 0.881]];

        let mut sigmoid = Sigmoid::new();
        let input = array![[-1.0, 0.0], [1.0, 2.0]];

        let actual = sigmoid.forward(&input);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e, 1e-3f64);
        }

        for (s, e) in sigmoid.out.iter().zip(expected.iter()) {
            assert_approx_eq!(s, e, 1e-3f64);
        }
    }
    #[test]
    fn test_sigmoid_backward() {
        let expected = array![[0.0, -12.0], [-42.0, -96.0]];

        let mut sigmoid = Sigmoid::new();
        sigmoid.out = array![[1.0, 2.0], [3.0, 4.0]];
        let input = array![[5.0, 6.0], [7.0, 8.0]];
        let actual = sigmoid.backward(&input);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e, 1e-3f64);
        }
    }
}
