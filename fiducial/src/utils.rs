use itertools::Itertools;

use slog::FnValue;

use ndarray::Array2;
use ndarray_stats::QuantileExt;
use slog::o;
use slog::Drain;
use slog::Logger;
use std::iter::zip;

use crate::fiducial_detector::FidDetectionParameter;

#[derive(Clone, Debug, Copy)]
///Define a Rectangle
pub struct Region {
    pub upper: (i32, i32),
    pub lower: (i32, i32),
}

pub(crate) fn filter_fid_groups<T: AsRef<[usize]>>(
    fid_center_components: &[T],
    fid_center_list: &[(f32, [f32; 2])],
    link_center_list: &[[f32; 2]],
) -> Vec<Vec<usize>> {
    let outlier_threshold = 0.2;
    let fid_radius_list = fid_center_list.iter().map(|(r, _y)| *r).collect_vec();
    fid_center_components
        .iter()
        .map(std::convert::AsRef::as_ref)
        .enumerate()
        .map(|(i, group)| {
            let current_links = link_center_list
                .iter()
                .enumerate()
                .filter(|(idx, _)| group.contains(idx))
                .map(|(_, item)| item)
                .collect_vec();
            let link_radius_list = current_links
                .iter()
                .map(|item| {
                    ((item[0] - fid_center_list[i].0).powi(2)
                        + (item[1] - fid_center_list[i].0).powi(2))
                    .sqrt()
                })
                .collect_vec();
            let outliers = link_radius_list
                .iter()
                .map(|x| {
                    ((*x - fid_radius_list[i]) / fid_radius_list[i]).abs() >= outlier_threshold
                })
                .collect_vec();
            zip(group, outliers)
                .filter(|(_a, b)| *b)
                .map(|(a, _b)| *a)
                .collect_vec()
        })
        .collect_vec()
}

///Separate edge links to individual fiducial.
pub(crate) fn separate_fid_group(
    fid_center_list: &[(f32, [f32; 2])],
    link_center_list: &[[f32; 2]],
    param: &FidDetectionParameter,
) -> Vec<Vec<usize>> {
    let data = fid_center_list
        .iter()
        .map(|&(_, x)| [x[0] as i32, x[1] as i32])
        .collect_vec();

    let kd_tree = kd_tree::KdIndexTree::build(&data);

    let mut fid_center_components = Vec::<Vec<usize>>::new();

    (0..fid_center_list.len()).for_each(|_| fid_center_components.push(Vec::<usize>::new()));

    link_center_list.iter().enumerate().for_each(|(i, pt)| {
        let found = kd_tree.nearest(&[pt[0] as i32, pt[1] as i32]);
        if let Some(res) = found {
            let est_center = fid_center_list[*(res.item)].1;
            //save only the points close to the center

            if pt_dist(&est_center, pt) < param.fid_region as f32 {
                fid_center_components[*(res.item)].push(i);
            }
        };
    });

    // remove outliers
    filter_fid_groups(&fid_center_components, fid_center_list, link_center_list)
}

///Calculate point-to-point distance."""
pub(crate) fn pt_dist(pt1: &[f32; 2], pt2: &[f32; 2]) -> f32 {
    ((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])).sqrt()
}

///Calculate angle of vector from x axis
pub(crate) fn pt_angle(pt1: &[f32; 2], pt2: &[f32; 2]) -> f32 {
    let (x, y) = ((pt1[0] - pt2[0]) as f64, (pt1[1] - pt2[1]) as f64);
    f64::atan2(y, x) as f32
}

fn dfs(data: &Array2<i32>, labeled: &mut Array2<i32>, x: usize, y: usize, c: i32, size: i32) {
    let mut node_stack = vec![(x, y)];

    while let Some(current_node) = node_stack.pop() {
        labeled[current_node] = c;
        for i in -size..size {
            for j in -size..size {
                if j == 0 && i == 0 {
                    continue;
                }
                let (nx, ny) = ((current_node.0 as i32 + i), (current_node.1 as i32 + j));

                if nx < 0
                    || nx > (data.dim().0 - 1) as i32
                    || ny < 0
                    || ny > (data.dim().1 - 1) as i32
                {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if data[(nx, ny)] == 1 && labeled[(nx, ny)] == 0 {
                    node_stack.push((nx, ny));
                }
            }
        }
    }
}

pub(crate) fn image_connected_components(
    data: &Array2<i32>,
    range: i32,
) -> Vec<((usize, usize), usize)> {
    let mut set = 1;
    let mut labeled = Array2::<i32>::from_elem(data.dim(), 0);

    for i in 1..data.dim().0 {
        for j in 1..data.dim().1 {
            if data[(i, j)] == 1 && labeled[(i, j)] == 0 {
                dfs(data, &mut labeled, i, j, set, range);
                set += 1;
            }
        }
    }

    let labeled_idx = labeled
        .indexed_iter()
        .filter(|(_, &val)| val != 0)
        .map(|((i, j), &val)| ((i, j), val as usize))
        .collect_vec();

    labeled_idx
}

pub struct ImageUtil {}
impl ImageUtil {
    /// Convert back to an image
    /// TODO: converting image to array and back rotates counterclockwise by 90 degrees
    pub(crate) fn to_gray_image(edges: &Array2<f32>) -> image::GrayImage {
        let (width, height) = edges.dim();
        edges.mapv(f32::abs);
        let max = *edges.max().unwrap();
        let min = *edges.min().unwrap();
        let data = edges
            .iter()
            .map(|&x| ((x - min) * 255.0 / (max - min)).round() as u8)
            .collect_vec();
        image::GrayImage::from_vec(height as u32, width as u32, data).unwrap()
    }

    #[allow(dead_code)]
    pub fn read_r_channel(path: &str) -> Array2<f32> {
        let image = image::open(path)
            .unwrap_or_else(|e| panic!("{}: {}", path, e))
            .into_rgb8();
        let raw = image.into_flat_samples();
        let step = raw.layout.width_stride;
        let red_channel: Vec<_> = raw
            .samples
            .iter()
            //.skip(2)
            .step_by(step)
            .copied()
            .collect();
        let channel_max = *red_channel.iter().max().unwrap() as f32;
        let red_channel = red_channel
            .iter()
            .map(|x| *x as f32 / channel_max)
            .collect();
        let red_image: Array2<f32> = Array2::from_shape_vec(
            (raw.layout.height as usize, raw.layout.width as usize),
            red_channel,
        )
        .unwrap();
        red_image
    }

    #[allow(dead_code)]
    pub fn resize_image(path: &str, ratio: f32, new_name: &str) {
        let image = image::open(path).unwrap().grayscale();
        let (w, h) = image.to_luma8().dimensions();
        let image = image.resize(
            (w as f32 / ratio) as u32,
            (h as f32 / ratio) as u32,
            image::imageops::FilterType::Nearest,
        );
        let _ = image.save(new_name);
    }
}

pub fn clamp(input: f32, min: f32, max: f32) -> f32 {
    debug_assert!(min <= max, "min must be less than or equal to max");
    if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    }
}

use std::sync::Once;
#[allow(dead_code)]
static INIT: Once = Once::new();

#[allow(dead_code)]
pub(crate) fn set_log_config() -> Logger {
    //let stdout: ConsoleAppender = ConsoleAppender::builder().encoder(Box::new(JsonEncoder::new())).build();
    // let stdout = ConsoleAppender::builder()
    //     .encoder(Box::new(PatternEncoder::new("{d} - {m}{n}")))
    //     .build();
    // let config = Config::builder()
    //     .appender(Appender::builder().build("stdout", Box::new(stdout)))
    //     .build(Root::builder().appender("stdout").build(LevelFilter::Info))
    //     .unwrap();
    // log4rs::init_config(config).unwrap();

    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::FullFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).build().fuse();
    //let drain = slog::Discard;
    //slog::Logger::root(drain, o!())
    Logger::root(
        drain,
        o!("place" =>
         FnValue(move |info| {
             format!("{}:{} {}",
                     info.file(),
                     info.line(),
                     info.module(),
                     )
         })
        ),
    )
}

#[cfg(test)]
mod tests {

    use super::{image_connected_components, ImageUtil};
    use crate::utils::set_log_config;
    use ndarray::arr2;
    use ndarray_npy::write_npy;
    use slog::info;

    #[test]
    fn test_image_connected_components() {
        let log = set_log_config();
        let graph = vec![
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ];

        let labeled = image_connected_components(&arr2(&graph), 1);

        info!(log, "n_components {:?}", labeled);
        assert_eq!(
            labeled,
            vec![((0, 1), 1), ((0, 2), 1), ((1, 2), 1), ((3, 4), 2)]
        );
    }

    #[test]
    #[ignore = "read/write test, please verify individually. NOTE: time consuming test"]
    fn test_resize_image() {
        ImageUtil::resize_image(
            "testing_data/single_fiducial.tif",
            2.,
            "testing_data/single_fiducial_resized.tif.png",
        )
    }

    #[ignore = "read/write test, please verify individually"]
    #[test]
    fn test_read_r_channel() {
        let path = "testing_data/single_fiducial.tif";
        let output = ImageUtil::read_r_channel(path);
        write_npy("testing_data/single_fiducial.npy", &output).unwrap();
    }
}
