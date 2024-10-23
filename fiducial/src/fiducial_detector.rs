use std::f32::consts::PI;
use std::time::Instant;

use crate::circle_fitting::circle_fitting;
use crate::edge_linking::edge_linking;
use crate::edge_subpixel::compute_edge_data;
use crate::encoding;
use crate::encoding::decode_fiducial;
use crate::encoding::TernaryCode;
use crate::find_fiducial::{find_fiducial, norm_direction};
use crate::utils::Region;
use crate::utils::{pt_dist, separate_fid_group, set_log_config};
use crate::IS_DEBUG;
use itertools::Itertools;
use ndarray::s;
use slog::error;

use ndarray::Array2;
use slog::info;

/// FidDetectionParameter   
#[derive(Clone, Debug)]
pub struct FidDetectionParameter {
    pub edge_threshold: f32,
    pub link_length: usize,
    pub hough_search_min: f32,
    pub hough_search_max: f32,
    pub hough_accu_threshold: f32,
    pub hough_group_range: i32,
    pub search_start: usize,
    pub search_end: usize,
    pub fid_region: i32,
    pub circle_fit_support: usize,
    pub circle_radius_low_limit: f32,
    pub circle_radius_up_limit: f32,
    pub circle_fit_var: f32,
    pub parallel_threshold: f32,
    pub outlier_threshold: f32,
    pub label: [f32; 3],
    pub resolution: f32,
    pub connected_range: i32,
    pub subpixel_threshold: f32,
}

impl Default for FidDetectionParameter {
    fn default() -> Self {
        Self {
            edge_threshold: Default::default(),
            subpixel_threshold: Default::default(),
            link_length: 6,
            hough_search_min: Default::default(),
            hough_search_max: Default::default(),
            hough_accu_threshold: 20.0,
            hough_group_range: Default::default(),
            search_start: Default::default(),
            search_end: Default::default(),
            fid_region: Default::default(),
            circle_fit_support: 10,
            circle_radius_low_limit: Default::default(),
            circle_radius_up_limit: Default::default(),
            circle_fit_var: 0.1,
            parallel_threshold: Default::default(),
            outlier_threshold: 0.4,
            resolution: 0.729,
            connected_range: 6,
            label: encoding::LABEL_HD,
        }
    }
}

impl FidDetectionParameter {
    #[allow(clippy::field_reassign_with_default)]
    pub fn new(
        search_start: usize,
        search_end: usize,
        edge_threshold: f32,
        parallel_threshold: f32,
    ) -> FidDetectionParameter {
        let mut ret = FidDetectionParameter::default();
        ret.search_start = search_start;
        ret.search_end = search_end;
        ret.edge_threshold = edge_threshold;
        ret.subpixel_threshold = edge_threshold;
        ret.parallel_threshold = parallel_threshold;
        ret.hough_search_min = 5.0 * (search_start as f32);
        ret.hough_search_max = 6.0 * (search_end as f32);
        ret.hough_group_range = (2 * search_end) as i32;
        ret.fid_region = (6 * search_end) as i32;
        ret.circle_radius_low_limit = 2.0 * (search_start as f32);
        ret.circle_radius_up_limit = 10.0 * (search_end as f32);
        ret
    }

    pub fn new_visium_hd_param() -> FidDetectionParameter {
        FidDetectionParameter::new(2, 10, 0.1, -0.9)
    }
}

impl FidDetectionParameter {
    pub fn new_py(
        search_start: usize,
        search_end: usize,
        edge_threshold: f32,
        parallel_threshold: f32,
        hough_accu_threshold: f32,
        circle_fit_support: usize,
        outlier_threshold: f32,
        label: Option<[f32; 3]>,
    ) -> Self {
        let mut param = FidDetectionParameter::new(
            search_start,
            search_end,
            edge_threshold,
            parallel_threshold,
        );
        param.hough_accu_threshold = hough_accu_threshold;
        param.circle_fit_support = circle_fit_support;
        param.outlier_threshold = outlier_threshold;
        if let Some(l) = label {
            param.label = l;
        }
        param
    }
}

/// Detect Visium HD fiducial.
pub fn detect_fiducial(
    img: &Array2<f32>,
    param: &FidDetectionParameter,
    region: &Option<Region>,
) -> Vec<(usize, [f32; 2])> {
    let log = set_log_config();

    let tick = Instant::now();
    if IS_DEBUG {
        info!(
            log,
            "\n\nget_image_link_data start {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }
    let (edge_data, links) = if let Some(region) = region {
        let img = &img
            .slice(s![
                region.upper.0..region.lower.0,
                region.upper.1..region.lower.1
            ])
            .to_owned();
        let edge_data = compute_edge_data(img, param);
        let links = edge_linking(&edge_data, param);
        (edge_data, links)
    } else {
        let edge_data = compute_edge_data(img, param);
        let links = edge_linking(&edge_data, param);
        (edge_data, links)
    };

    if IS_DEBUG {
        info!(
            log,
            "get_image_link_data took {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }

    let tick = Instant::now();

    if IS_DEBUG {
        info!(
            log,
            "\n\nfind_fiducial start {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }

    let est_fid_circles = find_fiducial(
        &links.link_centers,
        &norm_direction(&links.link_directions),
        edge_data.cols,
        edge_data.rows,
        param,
    );

    if IS_DEBUG {
        info!(
            log,
            "find_fiducial took {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }
    if IS_DEBUG && cfg!(debug_assertions) {
        info!(
            log,
            " est_fid_circles size {} {:?}",
            est_fid_circles.len(),
            est_fid_circles
        );
    }
    let tick = Instant::now();

    if IS_DEBUG {
        info!(
            log,
            "\n\nseparate_fid_group start {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }
    let fid_groups_list = separate_fid_group(&est_fid_circles, &links.link_centers, param);

    if IS_DEBUG {
        info!(
            log,
            "new_separate_fid_group took {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );

        info!(
            log,
            " fid_groups_list_size {:?}, fid_group_lens {:?}",
            fid_groups_list.len(),
            fid_groups_list.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }

    let tick = Instant::now();

    if IS_DEBUG {
        info!(
            log,
            "\n\ncompute_circles start {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }
    let (fid_centers, fid_radius) = compute_circles(
        &fid_groups_list,
        &est_fid_circles,
        param,
        &edge_data.pts,
        &links.link_list,
    );

    if IS_DEBUG {
        info!(
            log,
            "compute_circles took {:.3}s",
            tick.elapsed().as_millis() as f64 / 1000.0
        );

        info!(
            log,
            "fid_centers: {:?}, fid_radius: {:?}", fid_centers, fid_radius
        );
    }
    //move fiducial center coordinates from center based to corner based indexing
    let corner_based_fid_centers: Vec<_> = fid_centers
        .iter()
        .map(|pt| [pt[0] + 0.5, pt[1] + 0.5])
        .collect();
    std::iter::zip(
        decode_result(fid_radius, param, 5),
        corner_based_fid_centers,
    )
    .collect()
}

pub(crate) fn compute_circles(
    fid_groups_list: &[Vec<usize>],
    est_fid_centers_list: &[(f32, [f32; 2])],
    param: &FidDetectionParameter,
    edge_pt: &[[f32; 2]],
    link_list: &Array2<usize>,
) -> (Vec<[f32; 2]>, Vec<Vec<f32>>) {
    let log = set_log_config();

    let mut fid_center_list = Vec::<[f32; 2]>::new();
    let mut ring_radius_list = Vec::<Vec<f32>>::new();

    fid_groups_list
        .iter()
        .enumerate()
        .filter(|(_, link_group)| {
            if link_group.len() < param.circle_fit_support {
                error!(
                    log,
                    "link_group size  {} is less that required param.circle_fit_support  {}",
                    link_group.len(),
                    param.circle_fit_support
                );
            }
            link_group.len() >= param.circle_fit_support
        })
        .for_each(|(i, link_group)| {
            let est_center = est_fid_centers_list[i].1;
            let mut bad_ring = false;
            let mut num_pts = 0;

            let res = (0..param.link_length)
                .filter_map(|level| {
                    let pts =
                        get_level_pts(link_group, link_list, edge_pt, est_center, level, param);

                    //let pts_cleaned = remove_outliers(&pts, est_center, param);
                    num_pts = pts.len();
                    let (center, radius, bad_circle_fit) = circle_fitting(
                        &pts,
                        param.circle_radius_up_limit,
                        param.circle_radius_low_limit,
                        param.circle_fit_var,
                        PI,
                    );
                    if bad_circle_fit {
                        bad_ring = true;
                        None
                    } else {
                        Some((radius, center))
                    }
                })
                .collect_vec();

            if !bad_ring {
                let (radius_list, center_list): (Vec<f32>, Vec<[f32; 2]>) =
                    res.iter().cloned().unzip();

                let mut avg_center = center_list
                    .iter()
                    .fold([0.0_f32, 0.0], |acc, &pt| [acc[0] + pt[0], acc[1] + pt[1]]);

                avg_center[0] /= center_list.len() as f32;
                avg_center[1] /= center_list.len() as f32;

                fid_center_list.push(avg_center);
                ring_radius_list.push(radius_list);
            } else if IS_DEBUG {
                error!(log, "bad circle above, num points: {}", num_pts);
            }
        });

    (fid_center_list, ring_radius_list)
}

#[inline]
fn get_level_pts(
    link_group: &[usize],
    link_list: &Array2<usize>,
    edge_pt: &[[f32; 2]],
    est_center: [f32; 2],
    level: usize,
    param: &FidDetectionParameter,
) -> Vec<[f32; 2]> {
    link_group
        .iter()
        .map(|&idx| {
            let last = link_list.slice(s![idx, ..]).len() - 1;
            let start_pt = edge_pt[link_list[[idx, 0]]];
            let end_pt = edge_pt[link_list[[idx, last]]];
            let dist_start = pt_dist(&start_pt, &est_center);
            let dist_end = pt_dist(&end_pt, &est_center);

            if dist_end > dist_start {
                edge_pt[link_list[[idx, param.link_length - 1 - level]]]
            } else {
                edge_pt[link_list[[idx, level]]]
            }
        })
        .collect_vec()
}

pub(crate) fn decode_result(
    ring_radius_list: Vec<Vec<f32>>,
    param: &FidDetectionParameter,
    nbit: usize,
) -> Vec<usize> {
    //let log = set_log_config();
    ring_radius_list
        .iter()
        .filter(|ring_radii| ring_radii.len() == nbit + 1)
        .map(|ring_radii| {
            let a = &ring_radii[..ring_radii.len() - 1];
            let b = &ring_radii[1..];
            //subtract each elem from the next, normalize by first
            let thickness: Vec<_> = std::iter::zip(a, b)
                .map(|(p, q)| (p - q) / ring_radii[0])
                .collect();
            let ternary = decode_fiducial(&thickness, &param.label);
            TernaryCode::to_int(&ternary)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::utils::ImageUtil;

    use super::{decode_result, detect_fiducial, FidDetectionParameter};

    #[test]
    fn test_decode_result() {
        let param = FidDetectionParameter::new(2, 10, 0.1, -0.9);
        let ring_radius_list = vec![
            vec![
                35.419422, 28.992788, 22.613676, 18.144806, 11.848288, 7.29295,
            ],
            vec![
                35.483665, 28.991076, 22.578684, 18.238579, 14.1515, 11.312089,
            ],
            vec![
                35.476646, 29.016905, 22.845123, 20.154411, 16.062744, 11.626029,
            ],
            vec![
                35.407448, 28.938845, 22.785673, 20.341152, 18.219223, 15.645585,
            ],
            vec![
                35.51972, 31.366156, 28.68934, 24.704012, 20.186525, 13.856954,
            ],
        ];
        //let param = FidDetectionParameter::new(2.0, 10.0, 0.1, -0.9);
        let decoded = decode_result(ring_radius_list, &param, 5);
        let actual = [232, 228, 220, 216, 95];
        assert!(
            decoded.iter().eq(actual.iter()),
            "Expected {:?}, got {:?}",
            actual,
            decoded
        );
    }

    #[test]
    fn test_single_fiducial() {
        //Tests detection of individual fiducial
        let path = "testing_data/single_fiducial.tif";
        let param = FidDetectionParameter::new_visium_hd_param();
        let image = ImageUtil::read_r_channel(path);
        let actual_center = [42.722523, 43.691833];

        let (name, center) = detect_fiducial(&image, &param, &None)[0];
        assert_eq!(name, 36);
        assert!((actual_center[0] - center[0]).abs() < 0.01);
        assert!((actual_center[1] - center[1]).abs() < 0.01);
    }
}
