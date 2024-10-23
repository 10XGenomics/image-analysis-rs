use itertools::Itertools;
use ndarray::{arr2, s, Array2};
#[allow(unused_variables)]
/// Sobel vertical `South` gradient operator.
#[rustfmt::skip]
pub const SOBEL_SOUTH: [[f32; 3]; 3] = [
    [-1.0, -2.0, -1.0,],
    [0.0, 0.0, 0.0,],
    [1.0, 2.0, 1.0,],
];

/// Sobel horizontal `East` gradient operator.
#[rustfmt::skip]
pub const SOBEL_EAST: [[f32; 3]; 3] = [
    [-1.0, 0.0, 1.0,],
    [-2.0, 0.0, 2.0,],
    [-1.0, 0.0, 1.0,],
];

/// Sobel horizontal `West` gradient operator.
#[rustfmt::skip]
pub const SOBEL_WEST: [[f32; 3]; 3] = [
    [1.0, 0.0, -1.0,],
    [2.0, 0.0, -2.0,],
    [1.0, 0.0, -1.0,],
];
/// Sobel vertical `North` gradient operator.
#[rustfmt::skip]
pub const SOBEL_NORTH: [[f32; 3]; 3] = [
    [1.0, 2.0, 1.0,],
    [0.0, 0.0, 0.0,],
    [-1.0, -2.0, -1.0,],
];

pub fn spot_convolve_2d(image: &Array2<f32>, kernel: &Array2<f32>, i: usize, j: usize) -> f32 {
    let window = &image.slice(s![i - 1..i + 2, j - 1..j + 2]);
    (window * kernel).sum() / 4.0
}

pub fn sobel(image: &Array2<f32>, kernel: &[[f32; 3]; 3]) -> Array2<f32> {
    let kernel: Array2<f32> = arr2(kernel);
    let mut out = Array2::zeros(image.dim());
    (1..image.dim().0 - 1)
        .cartesian_product(1..image.dim().1 - 1)
        .for_each(|(i, j)| out[[i, j]] = spot_convolve_2d(image, &kernel, i, j));
    out
}

#[cfg(test)]
mod tests {

    use ndarray::Array2;
    use ndarray_linalg::assert_close_l1;
    use ndarray_npy::read_npy;

    use crate::utils::ImageUtil;

    use super::{sobel, SOBEL_NORTH};

    #[test]
    fn test_sobel() {
        let img: Array2<f32> = ImageUtil::read_r_channel("testing_data/single_fiducial.tif");
        let actual = sobel(&img, &SOBEL_NORTH);
        let expected: Array2<f32> = read_npy("testing_data/sobel_north.npy")
            .expect("Could not read expected sobel_north result");
        assert_close_l1!(&actual, &expected, 0.01);
    }
}
