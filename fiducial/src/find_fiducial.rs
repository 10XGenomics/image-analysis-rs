use std::{collections::HashSet, f32::consts::PI};

use itertools::Itertools;
use ndarray::{arr1, arr2, s, Array3, Axis};
use ndarray_npy::write_npy;
use ndarray_stats::QuantileExt;
use slog::info;

use crate::{
    bresenham::bresenham,
    edge_subpixel::write_vec,
    fiducial_detector::FidDetectionParameter,
    utils::{image_connected_components, set_log_config, ImageUtil},
    IS_DEBUG,
};

pub(crate) fn find_fiducial(
    feature_coord: &[[f32; 2]],
    norm_direction: &[[f32; 2]],
    n_rows: usize,
    n_cols: usize,
    param: &FidDetectionParameter,
) -> Vec<(f32, [f32; 2])> {
    let log = set_log_config();

    if cfg!(debug_assertions) && IS_DEBUG {
        write_vec("test_data/feature_coord.txt", feature_coord);
        write_vec("test_data/norm_direction.txt", norm_direction);
    }

    let accu_arr = circle_hough_accum(
        n_rows,
        n_cols,
        feature_coord,
        norm_direction,
        param.hough_search_min,
        param.hough_search_max,
    );

    if cfg!(debug_assertions) && IS_DEBUG {
        write_npy("test_data/accu_arr.npy", &accu_arr).unwrap();
    }

    let data = accu_arr.slice(s![0usize, .., ..]);
    let max = data.max().unwrap();
    let min = data.min().unwrap();

    if cfg!(debug_assertions) && IS_DEBUG {
        info!(
            log,
            "max {} min {}  thresh {}", max, min, param.hough_accu_threshold
        );
        let image = accu_arr.slice(s![0usize, .., ..]).mapv(|x| {
            if x >= param.hough_accu_threshold {
                x
            } else {
                0.0
            }
        });
        let _ = ImageUtil::to_gray_image(&image).save("image_accu_arr_init.png");
    }

    let image = accu_arr.slice(s![0usize, .., ..]).mapv(|x| {
        if x >= param.hough_accu_threshold {
            1_i32
        } else {
            0
        }
    });

    let labeled = image_connected_components(&image, param.connected_range);

    let (fid_center, fid_radius) = propose_circle(labeled, &accu_arr);
    let fid_radius = fid_radius
        .iter()
        .map(|&val| {
            param.hough_search_min
                + val * (param.hough_search_max - param.hough_search_min) / (2.0 * PI)
        })
        .collect_vec();

    let component_labels = group_pts(&fid_center, param.hough_group_range);

    component_labels
        .iter()
        .map(|component| {
            let centers = component.iter().map(|&idx| fid_center[idx]).collect_vec();
            let center = arr2(&centers).mean_axis(Axis(0)).unwrap();
            let radius = arr1(&fid_radius).mean().unwrap();
            (radius, [center[0], center[1]])
        })
        .collect_vec()
}

fn circle_hough_accum(
    n_rows: usize,
    n_cols: usize,
    feature_coord: &[[f32; 2]],
    norm_direction: &[[f32; 2]],
    min_radius: f32,
    max_radius: f32,
) -> Array3<f32> {
    let mut accu_arr = Array3::<f32>::from_elem((2, n_rows, n_cols), 0.0);
    let mut search_line = vec![[0; 2]; 3 * max_radius as usize + 1];

    let search_radius = max_radius - min_radius;
    let n_rows = accu_arr.dim().1 as i32;
    let n_cols = accu_arr.dim().2 as i32;

    for i in 0..(feature_coord.len()) {
        let (row_start, col_start) = (feature_coord[i][0] as i32, feature_coord[i][1] as i32);
        //search along both directions
        for sign in [1.0, -1.0] {
            let row_end = (sign * search_radius * norm_direction[i][0] + row_start as f32) as i32;
            let col_end = (sign * search_radius * norm_direction[i][1] + col_start as f32) as i32;
            let n_pts = bresenham(
                &mut search_line,
                max_radius as i32,
                (row_start, col_start),
                (row_end, col_end),
                true,
            );

            for [next_r, next_c] in search_line.iter().take(n_pts).copied() {
                if (next_r < 0) || (next_c < 0) || (next_r >= n_rows - 1) || (next_c >= n_cols - 1)
                {
                    break;
                }
                let theta =
                    2.0 * PI * (((next_r.pow(2) + next_c.pow(2)) as f32).sqrt() - min_radius)
                        / search_radius;

                //accumulate in complex space
                accu_arr[(0, next_r as usize, next_c as usize)] += theta.cos();
                accu_arr[(1, next_r as usize, next_c as usize)] += theta.sin();
            }
        }
    }

    //convert to amplitude and phase
    for i_1 in 1..(n_rows - 1) {
        for j_1 in 1..(n_cols - 1) {
            let (i, j) = (i_1 as usize, j_1 as usize);
            let real = accu_arr[(0, i, j)];
            let im = accu_arr[(1, i, j)];
            let amp = (real * real + im * im).sqrt();
            if amp > 0. {
                let sin_theta = im / amp;
                let cos_theta = real / amp;

                let theta = if sin_theta < 0. {
                    2.0 * PI - cos_theta.acos()
                } else {
                    cos_theta.acos()
                };
                accu_arr[(0, i, j)] = amp;
                accu_arr[(1, i, j)] = theta;
            } else {
                accu_arr[(0, i, j)] = 0.0;
                accu_arr[(1, i, j)] = 0.;
            }
        }
    }

    accu_arr
}

#[inline]
fn propose_circle(
    labeled: Vec<((usize, usize), usize)>,
    accu_arr: &Array3<f32>,
) -> (Vec<[f32; 2]>, Vec<f32>) {
    let labels: HashSet<_> = labeled.iter().map(|&x| x.1).collect();

    let mut circle_centers = vec![[0_f32; 2]; labels.len()];
    let mut circle_radius = vec![0_f32; labels.len()];
    let mut center_amp = vec![0.0; labels.len()];

    for ((i, j), label) in labeled {
        let l = label - 1;
        let amp = accu_arr[(0, i, j)];
        if amp > center_amp[l] {
            circle_centers[l][0] = i as f32;
            circle_centers[l][1] = j as f32;
            center_amp[l] = amp;
            circle_radius[l] = accu_arr[(1, i, j)]
        }
    }
    (circle_centers, circle_radius)
}

pub fn group_pts(pt_list: &[[f32; 2]], group_radius: i32) -> Vec<Vec<usize>> {
    let data = pt_list
        .iter()
        .map(|x| [x[0] as i32, x[1] as i32])
        .collect_vec();
    let kd_tree = kd_tree::KdIndexTree::build(&data);
    let mut processed = HashSet::<_>::new();

    let mut groups = Vec::<Vec<usize>>::new();
    pt_list.iter().enumerate().for_each(|(i, pt)| {
        if !processed.contains(&i) {
            let found = kd_tree.within_radius(&[pt[0] as i32, pt[1] as i32], group_radius);
            let mut neighbors = Vec::<usize>::new();
            let good_found = found
                .into_iter()
                .filter(|&idx| !processed.contains(idx))
                .copied()
                .collect_vec();
            good_found.iter().for_each(|&idx| {
                processed.insert(idx);
                neighbors.push(idx);
            });
            groups.push(neighbors);
        }
    });
    groups.into_iter().filter(|x| !x.is_empty()).collect_vec()
}

pub fn norm_direction(dirs: &[[f32; 2]]) -> Vec<[f32; 2]> {
    dirs.iter()
        .map(|d| {
            let mag = (d[0].powi(2) + d[1].powi(2)).sqrt();
            [d[0] / mag, d[1] / mag]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{
        edge_linking::edge_linking, edge_subpixel::compute_edge_data,
        fiducial_detector::FidDetectionParameter, utils::ImageUtil,
    };

    use super::{find_fiducial, norm_direction};

    #[test]
    fn test_find_fiducial() {
        let param = FidDetectionParameter::new_visium_hd_param();
        let image = ImageUtil::read_r_channel("testing_data/single_fiducial.tif");
        let edge_data = compute_edge_data(&image, &param);
        let links = edge_linking(&edge_data, &param);
        let est_fid_circles = find_fiducial(
            &links.link_centers,
            &norm_direction(&links.link_directions),
            edge_data.cols,
            edge_data.rows,
            &param,
        );
        let (_est_radius, est_center) = est_fid_circles[0];
        let actual_center = [42.722523, 43.691833];
        //let actual_radius = 5;
        assert!(
            (actual_center[1] - est_center[0]).abs() < 2.0
                && (actual_center[0] - est_center[1]).abs() < 2.0,
            "Estimated center is too far off  expected value"
        );
    }
}
