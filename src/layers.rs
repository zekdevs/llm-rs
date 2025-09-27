use anyhow::{Result, anyhow};

use crate::tensor::Tensor;

pub trait Layer {
    /// Executes the forward pass of the layer on the GPU.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationKind {
    Relu,
}

pub struct ActivationLayer {
    activation: ActivationKind,
}

impl ActivationLayer {
    pub fn new(activation: ActivationKind) -> Self {
        Self { activation }
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation {
            ActivationKind::Relu => input.relu(),
        }
    }
}

pub struct SoftmaxLayer;

impl SoftmaxLayer {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape().len() != 2 {
            return Err(anyhow!(
                "SoftmaxLayer expects input shaped [batch, features], got {:?}",
                input.shape()
            ));
        }
        input.softmax()
    }
}
