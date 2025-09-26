#![allow(dead_code)]

use anyhow::{Context, Result, anyhow};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use rand::prelude::*;
use std::usize;

/// A tensor stored on the GPU device.
///
/// # Fields
/// - `data`: The device buffer containing tensor elements.
/// - `device`: The CUDA device associated with this tensor.
/// - `dimentions`: The shape of the tensor (e.g., `[2, 3]` for a 2x3 matrix).
pub struct Tensor {
    pub data: DeviceBuffer<f32>,
    pub device: Device,
    pub shape: Vec<usize>, // shape
}

impl Tensor {
    /// Creates a new tensor on the specified device with the given dimensions.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor.
    /// * `device` - The CUDA device to allocate memory on.
    ///
    /// # Returns
    /// A `Result` containing the new `Tensor` or an error.
    pub fn new(shape: Vec<usize>, device: &Device) -> Result<Self> {
        let data_size = shape.iter().product::<usize>();
        let data = DeviceBuffer::zeroed(data_size)?;
        Ok(Self {
            data,
            device: device.clone(),
            shape,
        })
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Creates a tensor from host data, copying it to the device.
    pub fn from_host(data: Vec<f32>, shape: Vec<usize>, device: &Device) -> Result<Self> {
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(anyhow!(
                "Size mismatch: data has {} elements, but shape {:?} requires {}",
                data.len(),
                shape,
                expected_size
            ));
        }
        let mut tensor = Self::new(shape, device)?;
        tensor.copy_from_host(&data)?;
        Ok(tensor)
    }

    /// Copies data from host to device.
    pub fn copy_from_host(&mut self, host_data: &[f32]) -> Result<()> {
        let size = self.shape.iter().product::<usize>();

        // ensure data and tensor area are of the same size
        if host_data.len() != size {
            return Err(anyhow!(
                "Size mismatch: host_data has {} elements, but tensor shape {:?} requires {}",
                host_data.len(),
                self.shape,
                size
            ));
        }
        self.data.copy_from(host_data)?;
        Ok(())
    }

    /// Copies data from device to host
    pub fn to_host(&self) -> Result<Vec<f32>> {
        let size = self.shape.iter().product::<usize>();
        let mut host_data = vec![0.0f32; size];
        self.data.copy_to(&mut host_data[..])?;
        Ok(host_data)
    }

    /// Initializes a tensor to all ones.
    pub fn ones(shape: Vec<usize>, device: &Device) -> Result<Self> {
        let mut tensor = Self::new(shape.clone(), device)?;
        let host_data = vec![1.0f32; shape.iter().product::<usize>()];
        tensor.copy_from_host(&host_data)?;
        Ok(tensor)
    }

    /// Initializes a tensor filled with random floats
    pub fn randn(shape: Vec<usize>, device: &Device) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::rng();
        let host_data: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        Self::from_host(host_data, shape, device)
    }

    fn binary_elementwise(&self, other: &Tensor, kernel: KernelSpec) -> Result<Tensor> {
        ensure_same_device(&self.device, &other.device)?;

        let output_shape = broadcast_shapes(&self.shape, &other.shape)
            .with_context(|| format!("Broadcast failed for {}", kernel.human_name))?;

        let lhs_plan = BroadcastPlan::new(self, &output_shape)?;
        let rhs_plan = BroadcastPlan::new(other, &output_shape)?;
        let shape_buffer = DeviceBuffer::from_slice(&output_shape)?;

        let result = Tensor::new(output_shape.clone(), &self.device)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        let module = Module::from_ptx(kernel.ptx_source, &[])
            .with_context(|| format!("PTX load failed for {}", kernel.human_name))?;
        let function = module
            .get_function(kernel.function_name)
            .with_context(|| format!("Kernel load failed for {}", kernel.human_name))?;

        let n = result.size() as i32;
        let grid_size = ((n + 255) / 256) as u32;
        let block_size = 256u32;
        let rank = output_shape.len() as i32;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                self.data.as_device_ptr(),
                other.data.as_device_ptr(),
                result.data.as_device_ptr(),
                n,
                shape_buffer.as_device_ptr(),
                lhs_plan.device_strides.as_device_ptr(),
                rhs_plan.device_strides.as_device_ptr(),
                rank
            ))?;
        }

        stream
            .synchronize()
            .with_context(|| format!("Stream sync failed for {}", kernel.human_name))?;
        Ok(result)
    }

    /// Adds two tensors, supporting NumPy-style broadcasting.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_elementwise(
            other,
            KernelSpec::new("add", include_str!("add_kernel.ptx"), "add_kernel"),
        )
    }

    /// Backwards-compatible helper that forwards to [`Tensor::add`].
    pub fn add_broadcast(&self, other: &Tensor) -> Result<Tensor> {
        self.add(other)
    }

    /// Subtracts `other` from `self`, supporting broadcasting.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_elementwise(
            other,
            KernelSpec::new("sub", include_str!("sub_kernel.ptx"), "sub_kernel"),
        )
    }

    /// Multiplies two tensors elementwise, supporting broadcasting.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_elementwise(
            other,
            KernelSpec::new("mul", include_str!("mul_kernel.ptx"), "mul_kernel"),
        )
    }

    /// Divides `self` by `other` elementwise, supporting broadcasting.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_elementwise(
            other,
            KernelSpec::new("div", include_str!("div_kernel.ptx"), "div_kernel"),
        )
    }

    /// Adds a scalar to every element of the tensor.
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        let result = Tensor::new(self.shape.clone(), &self.device)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        let module = Module::from_ptx(include_str!("add_scalar_kernel.ptx"), &[])
            .context("PTX load failed for add_scalar")?;
        let function = module
            .get_function("add_scalar_kernel")
            .context("Kernel load failed for add_scalar")?;

        let n = self.size() as i32;
        let grid_size = ((n + 255) / 256) as u32;
        let block_size = 256u32;

        let input_ptr = self.data.as_device_ptr();
        let result_ptr = result.data.as_device_ptr();

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                input_ptr,
                scalar,
                result_ptr,
                n
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for add_scalar")?;
        Ok(result)
    }

    /// Subtracts a scalar from every element of the tensor.
    pub fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.add_scalar(-scalar)
    }

    /// Multiplies every element of the tensor by a scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = Tensor::from_host(vec![scalar], vec![1], &self.device)?;
        self.mul(&scalar_tensor)
    }

    /// Divides every element of the tensor by a scalar.
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        if scalar == 0.0 {
            return Err(anyhow!("Division by zero is not allowed"));
        }
        self.mul_scalar(1.0 / scalar)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let mut new_data =
            DeviceBuffer::zeroed(self.size()).expect("Failed to allocate device memory");
        new_data
            .copy_from(&self.data)
            .expect("Failed to copy device memory");
        Tensor {
            data: new_data,
            device: self.device,
            shape: self.shape.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct KernelSpec {
    human_name: &'static str,
    ptx_source: &'static str,
    function_name: &'static str,
}

impl KernelSpec {
    const fn new(
        human_name: &'static str,
        ptx_source: &'static str,
        function_name: &'static str,
    ) -> Self {
        Self {
            human_name,
            ptx_source,
            function_name,
        }
    }
}

struct BroadcastPlan {
    device_strides: DeviceBuffer<usize>,
}

impl BroadcastPlan {
    fn new(tensor: &Tensor, target_shape: &[usize]) -> Result<Self> {
        let aligned = compute_aligned_strides(&tensor.shape, target_shape)?;
        let device_strides = DeviceBuffer::from_slice(&aligned)?;
        Ok(Self { device_strides })
    }
}

fn ensure_same_device(lhs: &Device, rhs: &Device) -> Result<()> {
    if lhs.as_raw() != rhs.as_raw() {
        Err(anyhow!("Operands must reside on the same CUDA device"))
    } else {
        Ok(())
    }
}

fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let max_rank = lhs.len().max(rhs.len());
    let mut result = Vec::with_capacity(max_rank);

    for idx in 0..max_rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(idx + 1)
            .map(|pos| lhs[pos])
            .unwrap_or(1);
        let rhs_dim = rhs
            .len()
            .checked_sub(idx + 1)
            .map(|pos| rhs[pos])
            .unwrap_or(1);

        if lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1 {
            result.push(lhs_dim.max(rhs_dim));
        } else {
            return Err(anyhow!(
                "Incompatible dimensions for broadcasting: {} vs {}",
                lhs_dim,
                rhs_dim
            ));
        }
    }

    result.reverse();
    Ok(result)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn compute_aligned_strides(tensor_shape: &[usize], target_shape: &[usize]) -> Result<Vec<usize>> {
    let tensor_rank = tensor_shape.len();
    let target_rank = target_shape.len();
    let tensor_strides = compute_strides(tensor_shape);

    if target_rank == 0 {
        return Ok(Vec::new());
    }

    let mut aligned = vec![0usize; target_rank];

    for axis in 0..target_rank {
        let target_axis = target_rank - 1 - axis;
        let maybe_idx = tensor_rank.checked_sub(axis + 1);
        let (dim, stride) = match maybe_idx {
            Some(idx) => (tensor_shape[idx], tensor_strides[idx]),
            None => (1usize, 0usize),
        };

        let target_dim = target_shape[target_axis];
        if dim != target_dim && dim != 1 {
            return Err(anyhow!(
                "Broadcast violated compatibility at axis {}: {} vs {}",
                target_axis,
                dim,
                target_dim
            ));
        }

        aligned[target_axis] = if dim == 1 { 0 } else { stride };
    }

    Ok(aligned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_strides_matches_expected() {
        assert_eq!(compute_strides(&[4, 3, 2]), vec![6, 2, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert!(compute_strides(&[]).is_empty());
    }

    #[test]
    fn broadcast_shapes_handles_various_cases() {
        assert_eq!(broadcast_shapes(&[4, 1], &[1, 5]).unwrap(), vec![4, 5]);
        assert_eq!(
            broadcast_shapes(&[3, 1, 5], &[1, 7, 1]).unwrap(),
            vec![3, 7, 5]
        );
        assert_eq!(broadcast_shapes(&[2, 3], &[3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn compute_aligned_strides_handles_broadcasts() {
        assert_eq!(
            compute_aligned_strides(&[3, 1], &[3, 4]).unwrap(),
            vec![1, 0]
        );
        assert_eq!(
            compute_aligned_strides(&[1, 1, 1], &[2, 3, 4]).unwrap(),
            vec![0, 0, 0]
        );
        assert!(compute_aligned_strides(&[2, 2], &[3, 2]).is_err());
    }
}
