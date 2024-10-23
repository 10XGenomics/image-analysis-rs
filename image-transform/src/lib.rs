#![cfg_attr(all(test, feature = "unstable"), feature(test))]
#![allow(missing_docs)]
#![allow(unused_extern_crates)] //Otherwise intel_mkl_src is considered an unused crate

// Required for linking.
#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(not(target_os = "macos"))]
extern crate intel_mkl_src;

pub mod image_transformation;
pub const IS_DEBUG: bool = false;
