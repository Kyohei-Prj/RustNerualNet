use ndarray::{array, Array, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

mod layers;

use crate::layers::layers::*;

fn main() {
    let inputs: Array2<f64> = Array::random((1, 2), StandardNormal);
    println!("inputs: {:?}", inputs);
    println!();

    let mut relu = Relu::new();
    let after_relu = relu.forward(&inputs);
    println!("after relu: {:?}", after_relu);
    println!();

    let sample: Array2<f64> = array![[-5.0, -1.0, -0.5], [0.0, 0.5, 5.0]];
    let mut sigmoid = Sigmoid::new();
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

// struct Model {
//     pub layers: Vec<Box<dyn Layer>>,
// }

// impl Model {
//     pub fn new() -> Model {
// 	Model{layers: vec![]}
//     }
//     pub fn add(&mut self, layer: Box<dyn Layer>) {
// 	self.layers.push(layer)
//     }
//     pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
// 	let mut output = input.clone();
// 	for layer in &mut self.layers {
// 	    output = layer.forward(&output);
// 	}
// 	output
//     }
// }
