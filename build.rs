//! Build script for dinov3-ggml.
//!
//! Compiles the vendored GGML library (ggml-src submodule) plus the hand-written
//! ViT inference code (csrc/ggml_vit.c) into static libraries.
//!
//! Two build paths:
//!   - default : CPU-only, compiled directly with the `cc` crate (fast, no CMake)
//!   - `cuda`  : full CMake build of GGML with the CUDA backend (needs nvcc)
//!
//! All link directives are emitted from this crate. Do not move them into a
//! helper/sys crate: rustc drops native libs attached to crates that are never
//! referenced by Rust code.

use std::path::{Path, PathBuf};

fn main() {
    let ggml_src = Path::new("ggml-src");
    if !ggml_src.join("src/ggml.c").exists() {
        panic!(
            "GGML source not found at {}. Initialize the submodule first:\n\
             \n    git submodule update --init --recursive\n",
            ggml_src.display()
        );
    }

    let cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();

    if cuda {
        build_ggml_cuda(ggml_src);
    } else {
        build_ggml_cpu(ggml_src);
    }

    build_ggml_vit(ggml_src, cuda);

    // System libraries required by GGML
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=synchronization");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/ggml_vit.c");
    println!("cargo:rerun-if-changed=include/ggml_vit.h");
    println!("cargo:rerun-if-changed=ggml-src/src");
    println!("cargo:rerun-if-changed=ggml-src/include");
}

/// Compile our ViT inference C code.
fn build_ggml_vit(ggml_src: &Path, cuda: bool) {
    let mut build = cc::Build::new();
    build
        .file("csrc/ggml_vit.c")
        .include("include")
        .include(ggml_src.join("include"))
        .include(ggml_src.join("src"))
        .include(ggml_src.join("src/ggml-cpu"))
        .define("GGML_USE_CPU", None)
        .warnings(false);

    if cuda {
        build.define("GGML_USE_CUDA", None);
    }

    if cfg!(target_env = "msvc") {
        build.includes(msvc_system_includes());
    }

    build.compile("ggml-vit");
}

/// CPU-only GGML build via the `cc` crate (base + CPU backend, x86 kernels).
fn build_ggml_cpu(ggml_src: &Path) {
    let base_c = ["src/ggml.c", "src/ggml-alloc.c", "src/ggml-quants.c"];
    let base_cpp = [
        "src/ggml.cpp",
        "src/ggml-backend.cpp",
        "src/ggml-backend-meta.cpp",
        "src/ggml-backend-reg.cpp",
        "src/ggml-backend-dl.cpp",
        "src/ggml-threading.cpp",
        "src/gguf.cpp",
    ];
    let cpu_c = ["src/ggml-cpu/ggml-cpu.c", "src/ggml-cpu/quants.c"];
    let cpu_cpp = [
        "src/ggml-cpu/ggml-cpu.cpp",
        "src/ggml-cpu/ops.cpp",
        "src/ggml-cpu/binary-ops.cpp",
        "src/ggml-cpu/unary-ops.cpp",
        "src/ggml-cpu/vec.cpp",
        "src/ggml-cpu/repack.cpp",
        "src/ggml-cpu/traits.cpp",
        "src/ggml-cpu/hbm.cpp",
        "src/ggml-cpu/amx/amx.cpp",
        "src/ggml-cpu/amx/mmq.cpp",
    ];
    let arch_c = ["src/ggml-cpu/arch/x86/quants.c"];
    let arch_cpp = ["src/ggml-cpu/arch/x86/repack.cpp"];

    let extra_includes = if cfg!(target_env = "msvc") {
        msvc_system_includes()
    } else {
        Vec::new()
    };

    let mut cc_build = cc::Build::new();
    cc_build
        .files(base_c.iter().map(|s| ggml_src.join(s)))
        .files(cpu_c.iter().map(|s| ggml_src.join(s)))
        .files(arch_c.iter().map(|s| ggml_src.join(s)))
        .include(ggml_src.join("include"))
        .include(ggml_src.join("src"))
        .include(ggml_src.join("src/ggml-cpu"))
        .includes(&extra_includes)
        .define("GGML_USE_CPU", None)
        .define("GGML_VERSION", "\"0.1.0\"")
        .define("GGML_COMMIT", "\"unknown\"")
        .warnings(false);
    tune_native(&mut cc_build);
    cc_build.compile("ggml-c");

    let mut cxx_build = cc::Build::new();
    cxx_build
        .cpp(true)
        .std("c++17")
        .files(base_cpp.iter().map(|s| ggml_src.join(s)))
        .files(cpu_cpp.iter().map(|s| ggml_src.join(s)))
        .files(arch_cpp.iter().map(|s| ggml_src.join(s)))
        .include(ggml_src.join("include"))
        .include(ggml_src.join("src"))
        .include(ggml_src.join("src/ggml-cpu"))
        .includes(&extra_includes)
        .define("GGML_USE_CPU", None)
        .define("GGML_VERSION", "\"0.1.0\"")
        .define("GGML_COMMIT", "\"unknown\"")
        .warnings(false);
    tune_native(&mut cxx_build);
    cxx_build.compile("ggml-cpp");
    // cc emits rustc-link-lib=static=ggml-c / ggml-cpp automatically
}

/// Full GGML build (base + CPU + CUDA backends) via CMake, as static libs.
fn build_ggml_cuda(ggml_src: &Path) {
    let mut cfg = cmake::Config::new(ggml_src);
    cfg.profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_CUDA", "ON")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        // Flash-attention kernels are not used by ggml_vit: big compile-time win
        .define("GGML_CUDA_FA", "OFF")
        .define("GGML_CUDA_FA_ALL_QUANTS", "OFF")
        // NCCL is not needed for single-GPU inference
        .define("GGML_CUDA_NCCL", "OFF");

    // Restrict to the local GPU architecture to keep nvcc compile times sane.
    // Override with GGML_CUDA_ARCHS when building for other/multiple GPUs.
    let archs = std::env::var("GGML_CUDA_ARCHS").unwrap_or_else(|_| detect_cuda_arch());
    cfg.define("CMAKE_CUDA_ARCHITECTURES", &archs);
    println!("cargo:rerun-if-env-changed=GGML_CUDA_ARCHS");

    let dst = cfg.build();

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-cuda");

    // CUDA runtime libraries
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_path = cuda_path.replace('\\', "/");
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cuda");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Detect the compute capability of GPU 0 via nvidia-smi, e.g. "8.9" -> "89".
fn detect_cuda_arch() -> String {
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(first) = s.lines().next() {
            let arch: String = first.trim().chars().filter(|c| c.is_ascii_digit()).collect();
            if !arch.is_empty() {
                return arch;
            }
        }
    }
    // Reasonable default covering Ampere/Ada if detection fails
    "80;86;89".to_string()
}

/// Always compile GGML with optimizations and SIMD, even in debug profile.
/// Unoptimized scalar GGML is ~10x slower, which makes debug runs unusable.
fn tune_native(build: &mut cc::Build) {
    build.opt_level(3);
    build.debug(false);
    if cfg!(target_arch = "x86_64") {
        if cfg!(target_env = "msvc") {
            // MSVC defines __AVX__/__AVX2__ via /arch, but not FMA/F16C/SSE3
            build.flag("/arch:AVX2");
            build.define("__FMA__", None);
            build.define("__F16C__", None);
            build.define("__SSE3__", None);
            build.define("__SSSE3__", None);
        } else {
            build.flag("-march=native");
        }
    }
}

/// Locate MSVC + Windows SDK system include dirs (stdbool.h, ucrt, um, shared).
/// Needed because cl.exe invoked by the cc crate may lack the INCLUDE env when
/// cargo is not run from a VS developer prompt.
fn msvc_system_includes() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();

    // 1. vswhere (authoritative when available)
    if let Ok(output) = std::process::Command::new("vswhere")
        .args(["-latest", "-property", "installationPath"])
        .output()
    {
        let vs_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !vs_dir.is_empty() {
            add_msvc_include(&vs_dir, &mut out);
        }
    }

    // 2. Scan well-known install roots across drives/editions/years
    if out.is_empty() {
        let roots = [
            "C:/Program Files/Microsoft Visual Studio",
            "C:/Program Files (x86)/Microsoft Visual Studio",
            "D:/Microsoft Visual Studio",
            "E:/Microsoft Visual Studio",
        ];
        let years = ["2026", "2022"];
        let editions = ["Community", "Professional", "Enterprise", "BuildTools"];
        'outer: for root in roots {
            for year in years {
                for edition in editions {
                    let prefix = format!("{}/{}/{}", root, year, edition);
                    add_msvc_include(&prefix, &mut out);
                    if !out.is_empty() {
                        break 'outer;
                    }
                }
            }
        }
    }

    // 3. Windows SDK (ucrt/shared/um)
    for kits_str in ["C:/Program Files (x86)/Windows Kits/10/Include", "D:/Windows Kits/10/Include"] {
        let kits = Path::new(kits_str);
        if !kits.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(kits) {
            let mut versions: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            versions.sort_by_key(|e| e.file_name());
            if let Some(latest) = versions.last() {
                for sub in ["ucrt", "shared", "um"] {
                    let p = latest.path().join(sub);
                    if p.exists() && !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }
        break;
    }

    out
}

fn add_msvc_include(vs_dir: &str, out: &mut Vec<PathBuf>) {
    let msvc_tools = Path::new(vs_dir).join("VC/Tools/MSVC");
    if let Ok(entries) = std::fs::read_dir(&msvc_tools) {
        let mut versions: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        versions.sort_by_key(|e| e.file_name());
        if let Some(latest) = versions.last() {
            let inc = latest.path().join("include");
            if inc.exists() && !out.contains(&inc) {
                out.push(inc);
            }
        }
    }
}
