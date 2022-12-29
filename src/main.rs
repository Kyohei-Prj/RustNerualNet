use std::fmt::Debug;

use ndarray::{Array, Array2, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand_distr::num_traits::Float;

fn main() {

    let inputs: Array2<f64> = Array::random((3, 3), StandardNormal);
    println!("inputs: {:?}", inputs);
    println!();

    let relu = Relu::new();
    let after_relu = relu.forward(&inputs);
    println!("after relu: {:?}", after_relu);
    println!();

    let sample: Array2<f64> = array![[-5.0, -1.0, -0.5],[0.0, 0.5, 5.0]];
    let sigmoid = Sigmoid::new();
    let after_sigmoid = sigmoid.forward(&sample);
    println!("after sigoid: {:?}", after_sigmoid);
    println!();

    // let fc = FCLayer::new(2, 3);
    // println!("first layer: {:?}", fc);
    // println!();

    // let fc2 = FCLayer::new(3, 4);
    // println!("second layer: {:?}", fc2);
    // println!();

    // let fc3 = FCLayer::new(4, 2);
    // println!("second layer: {:?}", fc3);
    // println!();

    // let output = fc.forward(&inputs);
    // println!("first layer output: {:?}", output);
    // println!();

    // let output = fc2.forward(&output);
    // println!("second layer output: {:?}", output);
    // println!();

    // let output = fc3.forward(&output);
    // println!("third layer output: {:?}", output);
    // println!();

    // let mut model = Model::new();
    // model.add(fc);
    // model.add(fc2);
    // model.add(fc3);
    // let output = model.forward(&inputs);
    // println!("{:?}", output);

}

struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new() -> Model {
	Model{layers: vec![]}
    }
    pub fn add(&mut self, layer: Box<dyn Layer>) {
	self.layers.push(layer)
    }
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
	let mut output = input.clone();
	for layer in &self.layers {
	    output = layer.forward(&output);
	}
	output
    }
}

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
    d_bias: Array2<f64>
}

impl FCLayer {
    pub fn new(input_size: usize, output_size: usize) -> Box<FCLayer> {
	let fc = FCLayer{
	    weights: Array::random((input_size, output_size), StandardNormal),
	    bias: Array::random((1, output_size), StandardNormal),
	    x: array![[0.0]],
	    d_weights: array![[0.0]],
	    d_bias: array![[0.0]]
	};
	Box::new(fc)
    }
}

impl Layer for FCLayer {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
	self.x = input.clone();
	input.dot(&self.weights) + &self.bias
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64>{
	dx = dout.dot(&self.weights.t());
	self.d_weights = self.x.t().dot(&dout);
	self.d_bias = dout.sum()
    }
}

#[derive(Debug)]
pub struct Relu {}

impl Relu {
    pub fn new() -> Box<Relu> {
	Box::new(Relu{})
    }
}

impl Layer for Relu {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
	input.mapv(|x| if x > 0.0 {x} else {0.0})
    }
    fn backward(&mut self, loss: f64) {
	println!("diff: {}", loss)
    }
}

#[derive(Debug)]
pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Box<Sigmoid> {
	Box::new(Sigmoid{})
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
	input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
    fn backward(&mut self, loss: f64) {
	println!("diff: {}", loss)
    }
}
