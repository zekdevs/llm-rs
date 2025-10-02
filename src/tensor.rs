use crate::kernel_cache;
use anyhow::{Context, Result, anyhow};
use cust::memory::DeviceBuffer;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use rand::prelude::*;
use std::mem::size_of;
use std::usize;

/// A tensor stored on the GPU device.
///
/// # Fields
/// - `data`: The device buffer containing tensor elements.
/// - `device`: The CUDA device associated with this tensor.
/// - `dimensions`: The shape of the tensor (e.g., `[2, 3]` for a 2x3 matrix).
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

    /// Returns a read-only view of the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reshapes the tensor without copying data, consuming the tensor in the process.
    ///
    /// # Arguments
    /// * `new_shape` - The desired shape. The total number of elements must stay the same.
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Result<Self> {
        let new_size = new_shape.iter().product::<usize>();
        if new_size != self.size() {
            return Err(anyhow!(
                "Cannot reshape tensor of size {} into shape {:?}",
                self.size(),
                new_shape
            ));
        }
        self.shape = new_shape;
        Ok(self)
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

    /// Performs a dense matrix multiplication between two 2D tensors.
    ///
    /// Expects the left operand to have shape `[m, k]` and the right operand to have shape `[k, n]`.
    /// Returns a tensor with shape `[m, n]`.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        ensure_same_device(&self.device, &other.device)?;

        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(anyhow!(
                "Matmul requires 2D tensors, got lhs shape {:?} and rhs shape {:?}",
                self.shape,
                other.shape
            ));
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let k_other = other.shape[0];
        let n = other.shape[1];

        if k != k_other {
            return Err(anyhow!(
                "Matmul dimension mismatch: lhs {:?} vs rhs {:?}",
                self.shape,
                other.shape
            ));
        }

        let limits_ok = [m, n, k].iter().all(|&dim| dim <= i32::MAX as usize);
        if !limits_ok {
            return Err(anyhow!("Matrix dimensions exceed supported range"));
        }

        let result = Tensor::new(vec![m, n], &self.device)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        let module = kernel_cache::module(include_str!("matmul_kernel.ptx"), "matmul")?;
        let function = module
            .get_function("matmul_kernel")
            .context("Kernel load failed for matmul")?;

        let block = (16u32, 16u32, 1u32);
        let grid_x = ((n as u32) + block.0 - 1) / block.0;
        let grid_y = ((m as u32) + block.1 - 1) / block.1;
        let grid = (grid_x, grid_y, 1u32);

        unsafe {
            launch!(function<<<grid, block, 0, stream>>>(
                self.data.as_device_ptr(),
                other.data.as_device_ptr(),
                result.data.as_device_ptr(),
                m as i32,
                n as i32,
                k as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for matmul")?;
        Ok(result)
    }

    /// Computes analytic gradients for a matrix multiplication.
    ///
    /// Given \(C = A \times B\) and an upstream gradient `grad_output = ∂L/∂C`,
    /// this returns `(∂L/∂A, ∂L/∂B)`.
    pub fn matmul_backward(
        lhs: &Tensor,
        rhs: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        ensure_same_device(&lhs.device, &rhs.device)?;
        ensure_same_device(&lhs.device, &grad_output.device)?;

        if lhs.shape.len() != 2 || rhs.shape.len() != 2 || grad_output.shape.len() != 2 {
            return Err(anyhow!(
                "matmul_backward expects 2D tensors, got lhs {:?}, rhs {:?}, grad {:?}",
                lhs.shape,
                rhs.shape,
                grad_output.shape()
            ));
        }

        let m = lhs.shape[0];
        let k = lhs.shape[1];
        let rhs_rows = rhs.shape[0];
        let n = rhs.shape[1];

        if rhs_rows != k {
            return Err(anyhow!(
                "matmul_backward dimension mismatch: lhs {:?} vs rhs {:?}",
                lhs.shape,
                rhs.shape()
            ));
        }

        if grad_output.shape()[0] != m || grad_output.shape()[1] != n {
            return Err(anyhow!(
                "matmul_backward grad_output shape {:?} incompatible with forward ({}, {})",
                grad_output.shape(),
                m,
                n
            ));
        }

        let rhs_t = rhs.transpose2d()?;
        let grad_lhs = grad_output.matmul(&rhs_t)?;

        let lhs_t = lhs.transpose2d()?;
        let grad_rhs = lhs_t.matmul(grad_output)?;

        Ok((grad_lhs, grad_rhs))
    }

    /// Performs a binary elementwise operation on two tensors with broadcasting support.
    ///
    /// This method handles broadcasting of tensor shapes, computes the output shape,
    /// prepares broadcast plans for both operands, loads the specified CUDA kernel, and launches it on the GPU.
    ///
    /// # Arguments
    /// * `other` - The other tensor to operate with.
    /// * `kernel` - The kernel specification containing PTX source and function name.
    ///
    /// # Returns
    /// A `Result` containing the resulting `Tensor` or an error.
    ///
    /// # Errors
    /// - If tensors are on different devices.
    /// - If broadcasting fails.
    /// - If CUDA operations fail.
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

        let module = kernel_cache::module(kernel.ptx_source, kernel.function_name)?;
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

    /// Adds two tensors, supporting broadcasting.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_elementwise(
            other,
            KernelSpec::new("add", include_str!("add_kernel.ptx"), "add_kernel"),
        )
    }

    /// Computes the squared L2 norm of the tensor by summing the squares of all elements.
    pub fn l2_norm_squared(&self) -> Result<f32> {
        let host = self.to_host()?;
        let sum = host.iter().map(|&value| value * value).sum();
        Ok(sum)
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

        let module = kernel_cache::module(include_str!("add_scalar_kernel.ptx"), "add_scalar")?;
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

    /// Applies the ReLU activation to every element of the tensor.
    pub fn relu(&self) -> Result<Tensor> {
        self.unary_elementwise(KernelSpec::new(
            "relu",
            include_str!("relu_kernel.ptx"),
            "relu_kernel",
        ))
    }

    /// Applies the exponential function elementwise.
    pub fn exp(&self) -> Result<Tensor> {
        self.unary_elementwise(KernelSpec::new(
            "exp",
            include_str!("exp_kernel.ptx"),
            "exp_kernel",
        ))
    }

    /// Applies a numerically stable softmax along the feature dimension of a 2D tensor.
    pub fn softmax(&self) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(anyhow!(
                "Softmax currently supports 2D tensors shaped [batch, features], but received {:?}",
                self.shape
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let row_max = self.rowwise_max(rows, cols)?;
        let shifted = self.sub(&row_max)?;
        let exponentiated = shifted.exp()?;
        let row_sum = exponentiated.rowwise_sum(rows, cols)?;
        let denom = row_sum.add_scalar(1e-12)?;
        exponentiated.div(&denom)
    }

    /// Computes the gradient of a row-wise softmax given the upstream gradient and the softmax output.
    pub fn softmax_backward(probs: &Tensor, grad_output: &Tensor) -> Result<Tensor> {
        ensure_same_device(&probs.device, &grad_output.device)?;

        if probs.shape.len() != 2 {
            return Err(anyhow!(
                "softmax_backward expects 2D probability tensor, got {:?}",
                probs.shape
            ));
        }
        if grad_output.shape() != probs.shape() {
            return Err(anyhow!(
                "softmax_backward gradient shape {:?} must match probs shape {:?}",
                grad_output.shape(),
                probs.shape()
            ));
        }

        let dot = grad_output.mul(probs)?;
        let sum = dot.sum_rows()?; // shape [batch, 1]
        let adjusted = grad_output.sub(&sum)?;
        adjusted.mul(probs)
    }

    fn unary_elementwise(&self, kernel: KernelSpec) -> Result<Tensor> {
        let result = Tensor::new(self.shape.clone(), &self.device)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        let module = kernel_cache::module(kernel.ptx_source, kernel.function_name)?;
        let function = module
            .get_function(kernel.function_name)
            .with_context(|| format!("Kernel load failed for {}", kernel.human_name))?;

        let n = self.size() as i32;
        let grid_size = ((n + 255) / 256) as u32;
        let block_size = 256u32;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                self.data.as_device_ptr(),
                result.data.as_device_ptr(),
                n
            ))?;
        }

        stream
            .synchronize()
            .with_context(|| format!("Stream sync failed for {}", kernel.human_name))?;
        Ok(result)
    }

    fn rowwise_max(&self, rows: usize, cols: usize) -> Result<Tensor> {
        self.rowwise_reduce(
            KernelSpec::new(
                "rowwise max",
                include_str!("max_reduce_kernel.ptx"),
                "max_reduce_kernel",
            ),
            rows,
            cols,
        )
    }

    fn rowwise_sum(&self, rows: usize, cols: usize) -> Result<Tensor> {
        self.rowwise_reduce(
            KernelSpec::new(
                "rowwise sum",
                include_str!("sum_reduce_kernel.ptx"),
                "sum_reduce_kernel",
            ),
            rows,
            cols,
        )
    }

    /// Transposes a 2D tensor.
    pub fn transpose2d(&self) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(anyhow!(
                "transpose2d expects a 2D tensor, but received shape {:?}",
                self.shape
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let result = Tensor::new(vec![cols, rows], &self.device)?;
        let block = (16u32, 16u32, 1u32);
        let grid_x = ((cols as u32) + block.0 - 1) / block.0;
        let grid_y = ((rows as u32) + block.1 - 1) / block.1;
        let grid = (grid_x, grid_y, 1u32);

        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(include_str!("transpose_kernel.ptx"), "transpose2d")?;
        let function = module
            .get_function("transpose2d_kernel")
            .context("Kernel load failed for transpose2d")?;

        unsafe {
            launch!(function<<<grid, block, 0, stream>>>(
                self.data.as_device_ptr(),
                result.data.as_device_ptr(),
                rows as i32,
                cols as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for transpose2d")?;

        Ok(result)
    }

    /// Computes the sum across the last dimension for each row of a 2D tensor.
    pub fn sum_rows(&self) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(anyhow!(
                "sum_rows expects a 2D tensor, but received shape {:?}",
                self.shape
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        self.rowwise_sum(rows, cols)
    }

    /// Computes the sum across the first dimension for each column of a 2D tensor.
    pub fn sum_columns(&self) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(anyhow!(
                "sum_columns expects a 2D tensor, but received shape {:?}",
                self.shape
            ));
        }

        let cols = self.shape[1];
        let transposed = self.transpose2d()?;
        let summed = transposed.sum_rows()?;
        summed.reshape(vec![1, cols])
    }

    fn rowwise_reduce(&self, kernel: KernelSpec, rows: usize, cols: usize) -> Result<Tensor> {
        if rows == 0 || cols == 0 {
            return Err(anyhow!(
                "Row-wise reduction requires non-zero shape, got rows={}, cols={}",
                rows,
                cols
            ));
        }

        if self.shape.len() != 2 || self.shape[0] != rows || self.shape[1] != cols {
            return Err(anyhow!(
                "Row-wise reduction expects tensor shape [{}, {}], found {:?}",
                rows,
                cols,
                self.shape
            ));
        }

        let result = Tensor::new(vec![rows, 1], &self.device)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        let module = kernel_cache::module(kernel.ptx_source, kernel.function_name)?;
        let function = module
            .get_function(kernel.function_name)
            .with_context(|| format!("Kernel load failed for {}", kernel.human_name))?;

        let block_size = 256u32;
        let grid_size = rows as u32;
        let shared_bytes = (block_size as usize * size_of::<f32>()) as u32;

        unsafe {
            launch!(function<<<grid_size, block_size, shared_bytes, stream>>>(
                self.data.as_device_ptr(),
                result.data.as_device_ptr(),
                rows as i32,
                cols as i32
            ))?;
        }

        stream
            .synchronize()
            .with_context(|| format!("Stream sync failed for {}", kernel.human_name))?;
        Ok(result)
    }

    /// Backpropagates gradients through a ReLU activation.
    pub fn relu_backward(input: &Tensor, grad_output: &Tensor) -> Result<Tensor> {
        ensure_same_device(&input.device, &grad_output.device)?;

        if input.shape() != grad_output.shape() {
            return Err(anyhow!(
                "relu_backward shape mismatch: input {:?} vs grad {:?}",
                input.shape(),
                grad_output.shape()
            ));
        }

        let total = input.size();
        if total > i32::MAX as usize {
            return Err(anyhow!("relu_backward tensor too large for kernel launch"));
        }

        let result = Tensor::new(input.shape().to_vec(), &input.device)?;
        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;

        let module =
            kernel_cache::module(include_str!("relu_backward_kernel.ptx"), "relu_backward")?;
        let function = module
            .get_function("relu_backward_kernel")
            .context("Kernel load failed for relu_backward")?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                grad_output.data.as_device_ptr(),
                input.data.as_device_ptr(),
                result.data.as_device_ptr(),
                total as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for relu_backward")?;

        Ok(result)
    }
}

impl Clone for Tensor {
    /// Creates a deep copy of the tensor, including its data on the device.
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
    /// Creates a new `KernelSpec`.
    ///
    /// # Arguments
    /// * `human_name` - A human-readable name for the kernel.
    /// * `ptx_source` - The PTX source code as a string.
    /// * `function_name` - The name of the kernel function.
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

/// Plan for broadcasting a tensor to a target shape, containing device strides.
struct BroadcastPlan {
    device_strides: DeviceBuffer<usize>,
}

impl BroadcastPlan {
    /// Creates a new broadcast plan for the given tensor and target shape.
    ///
    /// # Arguments
    /// * `tensor` - The tensor to broadcast.
    /// * `target_shape` - The target shape to broadcast to.
    ///
    /// # Returns
    /// A `Result` containing the `BroadcastPlan` or an error.
    fn new(tensor: &Tensor, target_shape: &[usize]) -> Result<Self> {
        let aligned = compute_aligned_strides(&tensor.shape, target_shape)?;
        let device_strides = DeviceBuffer::from_slice(&aligned)?;
        Ok(Self { device_strides })
    }
}

/// Ensures that two devices are the same.
///
/// # Arguments
/// * `lhs` - The first device.
/// * `rhs` - The second device.
///
/// # Returns
/// A `Result` indicating success or an error if devices differ.
fn ensure_same_device(lhs: &Device, rhs: &Device) -> Result<()> {
    if lhs.as_raw() != rhs.as_raw() {
        Err(anyhow!("Operands must reside on the same CUDA device"))
    } else {
        Ok(())
    }
}

/// Computes the broadcasted shape of two tensors.
///
/// # Arguments
/// * `lhs` - The shape of the first tensor.
/// * `rhs` - The shape of the second tensor.
///
/// # Returns
/// A `Result` containing the broadcasted shape or an error if incompatible.
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

/// Computes the strides for a given shape.
///
/// Strides represent the number of elements to skip to move to the next dimension.
///
/// # Arguments
/// * `shape` - The shape of the tensor.
///
/// # Returns
/// A vector of strides.
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

/// Computes aligned strides for broadcasting a tensor shape to a target shape.
///
/// # Arguments
/// * `tensor_shape` - The original shape of the tensor.
/// * `target_shape` - The target shape to broadcast to.
///
/// # Returns
/// A `Result` containing the aligned strides or an error if broadcasting is invalid.
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
