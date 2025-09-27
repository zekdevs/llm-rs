use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("src");

    let kernels = [
        ("add_kernel.cu", "add_kernel.ptx"),
        ("sub_kernel.cu", "sub_kernel.ptx"),
        ("mul_kernel.cu", "mul_kernel.ptx"),
        ("div_kernel.cu", "div_kernel.ptx"),
        ("add_scalar_kernel.cu", "add_scalar_kernel.ptx"),
        ("relu_kernel.cu", "relu_kernel.ptx"), // Replaced gelu_kernel.cu
        ("max_reduce_kernel.cu", "max_reduce_kernel.ptx"),
        ("exp_kernel.cu", "exp_kernel.ptx"),
        ("sum_reduce_kernel.cu", "sum_reduce_kernel.ptx"),
    ];

    for (cu_file, ptx_file) in kernels {
        let ptx_path = Path::new(&out_dir).join(ptx_file);
        let src_ptx_path = src_dir.join(ptx_file);

        let status = Command::new("nvcc")
            .args(&[
                "-ptx",
                &format!("kernels/{}", cu_file),
                "-o",
                ptx_path.to_str().unwrap(),
            ])
            .status()
            .expect(&format!("nvcc failed for {}", cu_file));

        if !status.success() {
            panic!("nvcc compilation failed for {}", cu_file);
        }

        std::fs::copy(&ptx_path, &src_ptx_path).expect(&format!("Failed to copy {}", ptx_file));

        println!("cargo:rerun-if-changed=kernels/{}", cu_file);
    }
}
