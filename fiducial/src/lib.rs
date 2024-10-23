#![cfg_attr(all(test, feature = "unstable"), feature(test))]
#![allow(missing_docs)]
#![allow(unused_extern_crates)] //Otherwise intel_mkl_src is considered an unused crate

// Required for linking.
#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(not(target_os = "macos"))]
extern crate intel_mkl_src;

pub mod bresenham;
pub mod circle_fitting;
pub mod edge_linking;
pub mod edge_subpixel;
pub mod encoding;
pub mod fiducial_detector;
pub mod fiducial_registration;
pub mod filters;
pub mod find_fiducial;
pub mod utils;
pub const IS_DEBUG: bool = false;
