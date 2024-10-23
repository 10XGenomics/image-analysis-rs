use itertools::multizip;
use itertools::Itertools;
use ndarray::s;
use ndarray::Array2;
use ndarray::Zip;
use ndarray_npy::write_npy;
use std::fmt::Debug;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use crate::edge_linking::EdgeData;
use crate::fiducial_detector::FidDetectionParameter;
use crate::filters::{sobel, SOBEL_EAST, SOBEL_SOUTH};
use crate::IS_DEBUG;

pub(crate) fn write_vec<T>(p: &str, arr: &[T])
where
    T: Debug,
{
    let file = File::create(p).unwrap();
    let mut file = BufWriter::new(file);
    for value in arr {
        writeln!(file, "{:?}", value).unwrap();
    }
}

pub(crate) fn horizontal_subpixel_edge(
    image: &Array2<f32>,
    grad_x: &Array2<f32>,
    grad_y: &Array2<f32>,
    candidate_row: &[usize],
    candidate_col: &[usize],
    coef: &mut Array2<f32>,
) {
    /*Subpixel edge detection.

    This function is only for horizontal edges. To detect the vertical edges, the image can be rotated
    and then apply this function.

    Algorithm based on paper [1]
    MatLab code from author:
    https://www.mathworks.com/matlabcentral/fileexchange/48908-accurate-subpixel-edge-location

    Slide deck about the algorithm:
    http://serdis.dis.ulpgc.es/~atrujillo/subpixel/subpixelSlides.pdf

    [1] A. Trujillo-Pinoa, Accurate Subpixel Edge Location based on Partial Area Effect
        https://accedacris.ulpgc.es/bitstream/10553/43474/1/accurate_subpixel_edge_location.pdf

    Args:
        image: original image
        grad_x: gradient in the x direction
        grad_y: gradient in the y direction, NOTE: y direction points from small to large row count.
        candidate_row: row indices of edge pixels to consider
        candidate_col: col indices of edge pixels to consider
        coef: coefficients of the fitted polynomial, i.e., a, b, c, norm_x, and norm_y in that order
    */

    let n_rows = image.shape()[0];
    let n = coef.shape()[1];

    for i in 0..n {
        let row = candidate_row[i];
        let col = candidate_col[i];

        let mut m1 = -1;
        let mut m2 = 1;

        let (m, mut l1, mut l2, mut r1, mut r2, minl1, minr1, maxl2, maxr2) =
            if grad_x[[row, col]] * grad_y[[row, col]] >= 0.0 {
                (1, 0, 1, -1, 0, -3, -4, 4, 3)
            } else {
                (-1, -1, 0, 0, 1, -4, -3, 3, 4)
            };

        //Deal with close edges; algo in sec 5.1 of the paper
        if grad_x[[row, col]].abs() < 1.0 {
            (l1, l2, r1, r2) = (-1, 1, -1, 1);
        };
        while (l1 > minl1)
            && (grad_y[[row.checked_add_signed(l1).unwrap(), col - 1]].abs()
                >= grad_y[[row.checked_add_signed(l1).unwrap() - 1, col - 1]].abs())
        {
            l1 -= 1;
        }
        while (l2 < maxl2)
            && (row.checked_add_signed(l2).unwrap() + 1 < n_rows)
            && (grad_y[[row.checked_add_signed(l2).unwrap(), col - 1]].abs()
                >= grad_y[[row.checked_add_signed(l2).unwrap() + 1, col - 1]].abs())
        {
            l2 += 1;
        }
        while (m1 > -4)
            && (grad_y[[row.checked_add_signed(m1).unwrap(), col]].abs()
                >= grad_y[[row.checked_add_signed(m1).unwrap() - 1, col]].abs())
        {
            m1 -= 1;
        }
        while (m2 < 4) & (row + m2 + 1 < n_rows)
            && (grad_y[[row + m2, col]].abs() >= grad_y[[row + m2 + 1, col]].abs())
        {
            m2 += 1;
        }
        while (r1 > minr1)
            && (grad_y[[row.checked_add_signed(r1).unwrap(), col + 1]].abs()
                >= grad_y[[row.checked_add_signed(r1).unwrap() - 1, col + 1]].abs())
        {
            r1 -= 1;
        }
        while (r2 < maxr2)
            && (row.checked_add_signed(r2).unwrap() + 1 < n_rows)
            && (grad_y[[row.checked_add_signed(r2).unwrap(), col + 1]].abs()
                >= grad_y[[row.checked_add_signed(r2).unwrap() + 1, col + 1]].abs())
        {
            r2 += 1;
        }

        //take the extreme pixels as the high and low intensity
        let (hi_inten, low_inten) = if m > 0 {
            (
                (image[[row + m2, col]] + image[[row.checked_add_signed(r2).unwrap(), col + 1]])
                    / 2.0,
                (image[[row.checked_add_signed(l1).unwrap(), col - 1]]
                    + image[[row.checked_add_signed(m1).unwrap(), col]])
                    / 2.0,
            )
        } else {
            (
                (image[[row.checked_add_signed(l2).unwrap(), col - 1]] + image[[row + m2, col]])
                    / 2.0,
                (image[[row.checked_add_signed(m1).unwrap(), col]]
                    + image[[row.checked_add_signed(r1).unwrap(), col + 1]])
                    / 2.0,
            )
        };

        let (mut left_sum, mut mid_sum, mut right_sum) = (0.0, 0.0, 0.0);

        for j in l1..(l2 + 1) {
            left_sum += image[[row.checked_add_signed(j).unwrap(), col - 1]];
        }
        for j in m1..(m2 as isize + 1) {
            mid_sum += image[[row.checked_add_signed(j).unwrap(), col]];
        }
        for j in r1..(r2 + 1) {
            right_sum += image[[row.checked_add_signed(j).unwrap(), col + 1]];
        }

        let den = 2.0 * (hi_inten - low_inten);
        // equation 8 of [1], to calculate coefficient c, b, and a of second order polynomial
        coef[[2, i]] = (left_sum + right_sum - 2.0 * mid_sum
            + hi_inten * (2 * m2 as isize - l2 - r2) as f32
            - low_inten * (2 * m1 - l1 - r1) as f32)
            / den;
        coef[[1, i]] = (right_sum - left_sum + hi_inten * (l2 - r2) as f32
            - low_inten * (l1 - r1) as f32)
            / den;
        coef[[0, i]] =
            (2.0 * mid_sum - hi_inten * (1 + 2 * m2) as f32 - low_inten * (1 - 2 * m1) as f32)
                / den
                - coef[[2, i]] / 12.0;
        // equation 6 of [1], to calculate normal vector of the edge
        // note that y axis is flipped here so the new y axis points toward the row-increasing direction.
        coef[[3, i]] = (hi_inten - low_inten) * coef[[1, i]] / (1.0 + coef[[1, i]].powi(2)).sqrt();
        coef[[4, i]] = (hi_inten - low_inten) / (1.0 + coef[[1, i]].powi(2)).sqrt();
    }
}

pub(crate) fn hori_subpix_edge(
    image: &Array2<f32>,
    threshold: f32,
) -> (Vec<usize>, Vec<usize>, Array2<f32>) {
    let grad_x = sobel(image, &SOBEL_EAST);
    let grad_y = sobel(image, &SOBEL_SOUTH);

    let grad = grad_x.map(|e| e * e) + grad_y.map(|e| e * e);
    let grad = grad.mapv_into(f32::sqrt);

    if cfg!(debug_assertions) && IS_DEBUG {
        write_npy("test_data/grad_x.npy", &grad_x).unwrap();
        write_npy("test_data/grad_y.npy", &grad_y).unwrap();
        write_npy("test_data/grad.npy", &grad).unwrap();
    }

    let mut edge: Array2<_> = grad.map(|e| *e > threshold);

    let mut hori_greater = Array2::default(edge.dim());
    Zip::from(&mut hori_greater)
        .and(&grad_x)
        .and(&grad_y)
        .for_each(|z, &x, &y| *z = y.abs() > x.abs());

    edge = edge & hori_greater;

    let (dim_x, dim_y) = edge.dim();

    let mut x_greater: Array2<_> = edge.map(|_| true); //FIXME: hack again
    Zip::from(x_greater.slice_mut(s![1..dim_x, 0..dim_y]))
        .and(grad_y.slice(s![1..dim_x, 0..dim_y]))
        .and(grad_y.slice(s![0..dim_x - 1, 0..dim_y]))
        .for_each(|z, &y1, &y2| *z = y1.abs() > y2.abs());

    edge = edge & x_greater;

    let mut x_less: Array2<_> = edge.map(|_| true); //FIXME: hack again
    Zip::from(x_less.slice_mut(s![0..dim_x - 1, ..]))
        .and(grad_y.slice(s![0..dim_x - 1, ..]))
        .and(grad_y.slice(s![1..dim_x, ..]))
        .for_each(|z, &y1, &y2| *z = y1.abs() > y2.abs());

    edge = edge & x_less;

    let (edge_row, edge_col): (Vec<usize>, Vec<usize>) = (4..dim_x - 4)
        .cartesian_product(4..dim_y - 4)
        .filter(|(x, y)| edge[[*x, *y]])
        .unzip();

    let mut h_coeff: Array2<f32> = Array2::zeros((5, edge_row.len()));
    horizontal_subpixel_edge(image, &grad_x, &grad_y, &edge_row, &edge_col, &mut h_coeff);
    if cfg!(debug_assertions) && IS_DEBUG {
        write_npy("test_data/h_coeff.npy", &h_coeff).unwrap();
    }
    (edge_row, edge_col, h_coeff)
}

fn apply_mask<T>(target: &mut [T], mask: &[bool])
where
    T: Default + Copy,
{
    target
        .iter_mut()
        .enumerate()
        .for_each(|(idx, val)| *val = if mask[idx] { *val } else { T::default() });
}

pub(crate) fn compute_edge_data(image: &Array2<f32>, param: &FidDetectionParameter) -> EdgeData {
    //subpixel edge detection
    let (mut row_hori, mut col_hori, mut h_coef) = hori_subpix_edge(image, param.edge_threshold);
    let (mut col_vert, mut row_vert, mut v_coef) =
        hori_subpix_edge(&image.clone().reversed_axes(), param.edge_threshold);

    //create masks
    let valid_hori_edge_mask: Vec<_> = h_coef
        .slice(s![0, ..])
        .iter()
        .map(|item| (*item).abs() <= 3.0)
        .collect();
    let valid_vert_edge_mask: Vec<_> = v_coef
        .slice(s![0, ..])
        .iter()
        .map(|item| (*item).abs() <= 3.0)
        .collect();

    //apply masks
    apply_mask(&mut row_hori, &valid_hori_edge_mask);
    apply_mask(&mut col_hori, &valid_hori_edge_mask);
    apply_mask(&mut row_vert, &valid_vert_edge_mask);
    apply_mask(&mut col_vert, &valid_vert_edge_mask);

    for i in 0..5 {
        v_coef
            .slice_mut(s![i, ..])
            .indexed_iter_mut()
            .for_each(|(idx, val): (usize, &mut f32)| {
                *val = if valid_vert_edge_mask[idx] { *val } else { 0.0 }
            });

        h_coef
            .slice_mut(s![i, ..])
            .indexed_iter_mut()
            .for_each(|(idx, val): (usize, &mut f32)| {
                *val = if valid_hori_edge_mask[idx] { *val } else { 0.0 }
            });
    }

    let hori_x: Vec<_> = col_hori.iter().map(|x| *x as f32).collect();
    let mut hori_y = vec![0.0; row_hori.len()];
    Zip::from(&mut hori_y)
        .and(&row_hori)
        .and(h_coef.slice(s![0, ..]))
        .for_each(|z, x, y| *z = *x as f32 - y);

    let mut vert_x = vec![0.0; row_vert.len()];
    Zip::from(&mut vert_x)
        .and(&col_vert)
        .and(v_coef.slice(s![0, ..]))
        .for_each(|z, x, y| *z = *x as f32 - y);
    let mut vert_y: Vec<_> = row_vert.iter().map(|x| *x as f32).collect();

    //pixel location of each sub-pixel edge
    row_vert.extend(row_hori);
    col_vert.extend(col_hori);
    let (edge_row, edge_col) = (row_vert, col_vert);

    //exact subpixel edge location
    vert_x.extend(hori_x);
    vert_y.extend(hori_y);
    let (edge_x, edge_y) = (vert_x, vert_y);

    //norm direction of the edge
    let mut edge_norm_x = v_coef.slice(s![4, ..]).to_vec();
    edge_norm_x.extend(h_coef.slice(s![3, ..]));
    let mut edge_norm_y = v_coef.slice(s![3, ..]).to_vec();
    edge_norm_y.extend(h_coef.slice(s![4, ..]));
    let mut norm_amp = vec![0.0; edge_norm_x.len()];
    for (norm, norm_x, norm_y) in multizip((&mut norm_amp, &mut edge_norm_x, &mut edge_norm_y)) {
        *norm = (norm_x.powi(2) + norm_y.powi(2)).sqrt();
        *norm_x /= *norm;
        *norm_y /= *norm;
    }

    if cfg!(debug_assertions) && IS_DEBUG {
        write_vec("test_data/rust_edge_x.txt", &edge_x);
        write_vec("test_data/rust_edge_y.txt", &edge_y);
        write_vec("test_data/rust_edge_norm_x.txt", &edge_norm_x);
        write_vec("test_data/rust_edge_norm_y.txt", &edge_norm_y);
        write_vec("test_data/edge_row", &edge_row);
        write_vec("test_data/edge_col", &edge_col);
    }

    let pixels = edge_row
        .into_iter()
        .zip(edge_col)
        .map(|(x, y)| [x, y])
        .collect_vec();
    let pts = edge_x
        .into_iter()
        .zip(edge_y)
        .map(|(x, y)| [x, y])
        .collect_vec();
    let norms = edge_norm_x
        .into_iter()
        .zip(edge_norm_y)
        .map(|(x, y)| [x, y])
        .collect_vec();

    EdgeData {
        pts,
        pixels,
        norms,
        rows: image.dim().0,
        cols: image.dim().1,
    }
}

#[cfg(test)]
mod tests {

    use core::f32;

    use crate::edge_subpixel::compute_edge_data;
    use crate::edge_subpixel::hori_subpix_edge;
    use crate::fiducial_detector::FidDetectionParameter;
    use crate::utils::ImageUtil;
    use ndarray::Array2;

    #[test]
    fn test_subpix() {
        let image: Array2<f32> = ImageUtil::read_r_channel("testing_data/single_fiducial.tif");
        let param = FidDetectionParameter::new(0, 0, 0.3, 0.3);
        let (edge_row, edge_col, _) = hori_subpix_edge(&image, param.edge_threshold);

        //sanity check
        assert!(
            edge_row.len() == edge_col.len(),
            "Did not get the same number of row and column positions"
        );
        assert!(
            edge_row.len() > 300,
            "Found fewer than 300 points in test image"
        );
        //Width of largest ring is roughly correct
        assert!(
            (edge_row.iter().max().unwrap() - edge_row.iter().min().unwrap()).abs_diff(72) < 2,
            "Detected pixels imply fiducial of incorrect width"
        );
    }

    #[test]
    fn test_compute_edge_data() {
        let image: Array2<f32> = ImageUtil::read_r_channel("testing_data/single_fiducial.tif");
        let param = FidDetectionParameter::new(0, 0, 0.3, 0.3);
        let edges = compute_edge_data(&image, &param);

        //sanity check
        assert!(
            edges.pts.len() == edges.pixels.len() && edges.pixels.len() == edges.norms.len(),
            "Edge data is inconsistent"
        );

        assert!(
            (edges.rows, edges.cols) == (86, 85),
            "Image size incorrectly read"
        );

        //basic pixel threshold
        assert!(
            edges.pixels.len() > 700,
            "Found {:?} edge pixels, expected at least 700",
            edges.pixels.len()
        );
    }
}
