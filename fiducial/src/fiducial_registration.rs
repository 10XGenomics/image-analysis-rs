use ndarray::{array, aview1, s, Array1, Array2};
use ndarray_linalg::{svd::SVD, LeastSquaresSvd};
use rand::seq::IteratorRandom;
use std::{collections::HashSet, iter::FromIterator, str::FromStr};

use crate::{
    fiducial_detector::{detect_fiducial, FidDetectionParameter},
    utils::ImageUtil,
};

#[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
pub struct Coord(pub f64, pub f64);

impl Coord {
    pub fn new(x: f64, y: f64) -> Coord {
        Coord(x, y)
    }
    pub fn slope(self, other: &Coord) -> f64 {
        (self.1 - other.1) / (self.0 - other.0)
    }
    pub fn from_arr(pts: &[[f64; 2]]) -> Vec<Coord> {
        pts.iter().map(|x| Coord(x[0], x[1])).collect()
    }
    pub fn sum(self, other: &Coord) -> Coord {
        Coord(self.0 + other.0, self.1 + other.1)
    }
    pub fn to_arr(self) -> [f64; 2] {
        [self.0, self.1]
    }
}

impl FromStr for Coord {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ss: Vec<&str> = s.split(", ").collect();
        let x = f64::from_str(&ss[0][1..]).unwrap();
        let y = f64::from_str(&ss[1][..ss[1].len() - 1]).unwrap();
        Ok(Coord(x, y))
    }
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct FiducialPoint {
    pub name: usize,
    pub center: Coord,
}

impl FromStr for FiducialPoint {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ss: Vec<&str> = s.split_whitespace().collect();
        let n: usize = usize::from_str(ss[0]).unwrap();
        let x: f64 = f64::from_str(ss[1]).unwrap();
        let y: f64 = f64::from_str(ss[2]).unwrap();
        Ok(FiducialPoint {
            name: n,
            center: Coord(x, y),
        })
    }
}

#[derive(Debug)]
pub struct FiducialFrame {
    pub points: Vec<FiducialPoint>,
}

impl FiducialFrame {
    pub fn new(points: Vec<FiducialPoint>) -> FiducialFrame {
        let frame = FiducialFrame { points };
        frame.validate();
        frame
    }
    pub fn from_points(names: &[usize], points: &[Coord]) -> FiducialFrame {
        let frame_vec: Vec<_> = names
            .iter()
            .zip(points)
            .map(|(n, c)| FiducialPoint {
                name: *n,
                center: c.to_owned(),
            })
            .collect();
        let frame = FiducialFrame { points: frame_vec };
        frame.validate();
        frame
    }
    pub fn names(&self) -> Vec<usize> {
        self.points.iter().map(|x| x.name).collect()
    }
    pub fn centers(&self) -> Vec<Coord> {
        self.points.iter().map(|x| x.center).collect()
    }
    pub fn overlap(&self, other: &FiducialFrame) -> HashSet<usize> {
        let my_fiducials: HashSet<usize> = HashSet::from_iter(self.names());
        let other_fiducials: HashSet<usize> = HashSet::from_iter(other.names());
        let overlap = my_fiducials.intersection(&other_fiducials);
        overlap.map(std::borrow::ToOwned::to_owned).collect()
    }
    pub fn subset(&mut self, names: &HashSet<usize>) -> Vec<Coord> {
        self.sort();
        let mut out = Vec::with_capacity(names.len());
        for fid_point in &self.points {
            if names.contains(&fid_point.name) {
                out.push(fid_point.center)
            }
        }
        out
    }
    pub fn sort(&mut self) {
        self.points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    pub fn validate(&self) {
        let n = self.names();
        let nsize = HashSet::<usize>::from_iter(n.iter().cloned()).len();
        assert!(nsize == n.len(), "non-unique names in Fiducial Frame");
    }
}

pub fn similarity_transform_components(src: &[Coord], target: &[Coord]) -> (f64, f64, f64, f64) {
    assert!(
        src.len() == target.len(),
        "similarity transform source and target coordinates must have same length"
    );
    assert!(src.len() >= 2, "too few points for similarity transform");
    let n = src.len();
    let mut target_data = Array1::zeros(2 * n);
    for i in 0..n {
        target_data[2 * i] = target[i].0;
        target_data[2 * i + 1] = target[i].1;
    }
    let mut moving_data = Array2::zeros((2 * n, 4));
    for i in 0..2 * n {
        moving_data[[i, 2 + i % 2]] = 1.0;
        let coord_index = i / 2;
        let (x, y) = (src[coord_index].0, src[coord_index].1);
        let (a, b) = if i % 2 == 0 { (x, -y) } else { (y, x) };
        moving_data[[i, 0]] = a;
        moving_data[[i, 1]] = b;
    }

    let lstsq = moving_data.least_squares(&target_data).unwrap().solution;
    (lstsq[0], lstsq[1], lstsq[2], lstsq[3])
}

pub fn similarity_transform(src: &[Coord], target: &[Coord]) -> Array2<f64> {
    let lstsq = similarity_transform_components(src, target);
    let mut res = Array2::zeros((3, 3));
    res[[0, 0]] = lstsq.0;
    res[[0, 1]] = -lstsq.1;
    res[[0, 2]] = lstsq.2;

    res[[1, 0]] = lstsq.1;
    res[[1, 1]] = lstsq.0;
    res[[1, 2]] = lstsq.3;

    res[[2, 2]] = 1.0;
    res
}

fn homography(src: &[Coord], target: &[Coord]) -> Option<Array2<f64>> {
    //Finds homography transformation from src to dest
    //Set up matrix for decomposition
    assert!(
        src.len() == target.len(),
        "homography source and target must have same length"
    );
    if src.len() < 4 {
        return None;
    }
    let mut h: Array2<f64> = Array2::zeros((2 * src.len(), 9));
    for (i, (p1, p2)) in src.iter().zip(target.iter()).enumerate() {
        let Coord(x, y) = *p1;
        let Coord(u, v) = *p2;

        let row1 = vec![x, y, 1., 0., 0., 0., -u * x, -u * y, -u];
        let row2 = vec![0., 0., 0., x, y, 1., -v * x, -v * y, -v];

        h.row_mut(2 * i).assign(&aview1(&row1));
        h.row_mut(2 * i + 1).assign(&aview1(&row2));
    }

    //Perform SVD to get homography matrix
    let (_u, _s, vt) = h.svd(false, true).expect("could not compute svd");
    let mut trans_mat = vt
        .expect("Could not compute svd")
        .row(8)
        .to_owned()
        .into_shape((3, 3))
        .expect("Not a length 9 vector");

    trans_mat /= trans_mat[[2, 2]];
    Some(trans_mat)
}

//Generic 3x3 point transform
pub fn transform_pts_2d(pts: &[Coord], transform: &Array2<f64>) -> Vec<Coord> {
    assert_eq!(transform.shape(), &[3, 3]);
    pts.iter()
        .map(|x| {
            let pt: Array1<f64> = array![x.0, x.1, 1.0];
            let mut pt = transform.dot(&pt);
            pt /= pt[2];
            Coord(pt[0], pt[1])
        })
        .collect()
}

//Computes MSE between src and target coordinates.
pub fn registration_error(src: &[Coord], target: &[Coord], transform: &Array2<f64>) -> Vec<f64> {
    assert!(
        src.len() == target.len(),
        "Input coordinate arrays must be of the same length"
    );
    let transformed = transform_pts_2d(src, transform);
    std::iter::zip(transformed, target)
        .map(|(x, y)| {
            let d = (x.0 - y.0, x.1 - y.1);
            (d.0 * d.0 + d.1 * d.1).sqrt()
        })
        .collect()
}

fn ransac(
    src: &[Coord],
    target: &[Coord],
    num_iter: usize,
    num_pts: usize,
    threshold: f64,
) -> Option<Array2<f64>> {
    /* RANSAC-like algorithm for detecting incorrectly decoded fiducials.

    Randomly samples num_pts detected fiducials and uses them to generate a transformation matrix.
    Evaluates the output based on the number of design fiducials with significant registration error
    given that matrix. Repeats this num_iter times and picks the transformation matrix with the lowest
    number of high-error fiducials. ASSUMES EQUAL NUMBER OF SRC AND TARGET POINTS, AND THAT THEY ARE
    IN THE SAME ORDER.

    Args:
        src: points in space to find transformation from.
        target: points in space to find transformation to.
        num_iter: Number of random samples to try.
        num_pts: Number of fiducials to use per sample.
        threshold: Threshold (in pixels) of registration error to classify as inlier.
     */
    let mut best_homography = Array2::zeros((3, 3));
    let mut best_error = src.len();
    let rng = &mut rand::thread_rng();
    for _ in 0..num_iter {
        let indices = (0..src.len()).choose_multiple(rng, num_pts);
        let src_sample: Vec<Coord> = indices.iter().map(|i| src[*i]).collect();
        let target_sample: Vec<Coord> = indices.iter().map(|i| target[*i]).collect();
        let candidate_homography = homography(&src_sample, &target_sample);
        if let Some(x) = candidate_homography {
            let error = registration_error(src, target, &x)
                .into_iter()
                .filter(|x| *x > threshold)
                .count();
            if error < best_error {
                best_error = error;
                best_homography = x;
            }
        }
    }

    //use best homography to identify valid points:
    let valid_indices: Vec<_> = registration_error(src, target, &best_homography)
        .iter()
        .enumerate()
        .filter(|(_x, y)| **y < threshold)
        .map(|(x, _y)| x)
        .collect();

    let src_sample: Vec<Coord> = valid_indices.iter().map(|i| src[*i]).collect();
    let target_sample: Vec<Coord> = valid_indices.iter().map(|i| target[*i]).collect();
    homography(&src_sample, &target_sample)
}

pub fn fiducial_registration(
    detect: &mut FiducialFrame,
    design: &mut FiducialFrame,
) -> Option<(Array2<f64>, usize)> {
    //Fiducial Registration between two sets of fiducials using the RANSAC method
    let overlap = detect.overlap(design);
    let detect_subset = detect.subset(&overlap);
    let design_subset = design.subset(&overlap);
    let homography = ransac(&design_subset, &detect_subset, 10, 10, 1.0);
    let h = match homography {
        Some(h) => h,
        None => return None,
    };
    let transformed = transform_pts_2d(&design_subset, &h);

    let correct = (0..overlap.len())
        .map(|i| {
            let (x, y) = (detect_subset[i], transformed[i]);
            let d = (x.0 - y.0, x.1 - y.1);

            (d.0 * d.0 + d.1 * d.1).sqrt()
        })
        .filter(|e| *e < 1.0)
        .count();
    Some((h, correct))
}

pub fn isolate_perspective_transform(
    detect: &mut FiducialFrame,
    design: &mut FiducialFrame,
) -> Option<Array2<f64>> {
    /* Attempts to isolate the perspective transform between two fiducial frames,
      removing the scaling and 2D rotation components. Returns a 3x3 transformation
      matrix.

      Args:
          detect: The detected fiducials in the cytassist image
          design: Fiducials from the .slide design file
          hmat: The full perspective transform between design and detect as
                computed by fiducial_registration
    */
    let overlap = detect.overlap(design);
    let detect_subset = detect.subset(&overlap);
    let design_subset = design.subset(&overlap);
    _isolate_perspective_transform(&design_subset, &detect_subset)
}

fn _isolate_perspective_transform(src: &[Coord], target: &[Coord]) -> Option<Array2<f64>> {
    // Approximation of "perpective-only" transform, removing scaling and
    // rotation components.
    // black_box required because similarity_transform call seems to be
    // erroneously optimized out by the compiler otherwise.
    let sim_mat = std::hint::black_box(similarity_transform(src, target));
    let h_inv = homography(target, src);
    h_inv.map(|hinv| sim_mat.dot(&hinv))
}

fn refine_individual(
    image: &Array2<f32>,
    param: &FidDetectionParameter,
    bbox_width: f64,
) -> Option<Coord> {
    let fiducial = detect_fiducial(image, param, &None);
    if fiducial.len() != 1 {
        return None;
    }
    let point = fiducial[0].1;
    Some(Coord(
        point[0] as f64 - bbox_width,
        point[1] as f64 - bbox_width,
    ))
}

pub fn fiducial_refinement(
    image_path: &str,
    manual_alignment: &Array2<f64>,
    design: &mut FiducialFrame,
) -> Option<(Array2<f64>, FiducialFrame)> {
    /* Refines a manual alignment if possible. Returns None if no refinement can be found.
      Uses the manual alignment to identify bounding boxes for each fiducial, then runs
      detect_fiducial to identify individual fiducials in each box. If at least
      20 fiducials are found, output the fiducial_registration transformation matrix,
      otherwise return None

      Args:
          image_path: Path to the cytassist image
          manual_alignment: A 3x3 similarity transform from the design space to the image space
          design: A fiducial frame consisting of fiducials found in the .slide file

    */
    //Apply manual transform to design
    let image = ImageUtil::read_r_channel(image_path);
    let design_centers = design.centers();
    let manual_fiducial_centers = transform_pts_2d(&design_centers, manual_alignment);
    let manual_frame = FiducialFrame::from_points(&design.names(), &manual_fiducial_centers);

    //Isolate individual fiducials
    let param = FidDetectionParameter::new_visium_hd_param();
    let iso_detections: Vec<_> = manual_frame
        .points
        .iter()
        .filter_map(|pt| {
            let center = pt.center;
            let name = pt.name;
            let (x, y) = (center.1.round() as usize, center.0.round() as usize); //swap ordering due to change of axis notation
            let bbox_width = 50;
            let single_fid = image
                .slice(s![
                    x - bbox_width..x + bbox_width,
                    y - bbox_width..y + bbox_width
                ])
                .to_owned();
            let offset = refine_individual(&single_fid, &param, bbox_width as f64);
            offset.map(|refined_pt| FiducialPoint {
                name,
                center: refined_pt.sum(&Coord(y as f64, x as f64)),
            })
        })
        .collect();

    //identify fiducial, use decoding from design
    let mut iso_detections = FiducialFrame {
        points: iso_detections,
    };

    //if less than 20 fiducials, default to original manual alignment
    if iso_detections.points.len() < 20 {
        return None;
    };
    if let Some(reg) = fiducial_registration(&mut iso_detections, design) {
        return Some((reg.0, iso_detections));
    }
    None
}

pub fn fiducial_refinement_and_perspective_transform_isolation(
    image_path: &str,
    manual_alignment: &Array2<f64>,
    design: &mut FiducialFrame,
) -> Option<(Array2<f64>, Array2<f64>)> {
    /* Convenience function to combine fiducial refinement and perspective transform isolation.
       Outputs (perspective transform, isolated tilt transform)

       Args:
          image_path: Path to the cytassist image
          manual_alignment: A 3x3 similarity transform from the design space to the image space
          design: A fiducial frame consisting of fiducials found in the .slide file

    */
    let refinement = fiducial_refinement(image_path, manual_alignment, design);
    if let Some((hmat, mut detect)) = refinement {
        let p = isolate_perspective_transform(&mut detect, design);
        if let Some(perspective_transform) = p {
            return Some((hmat, perspective_transform));
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use assert_approx_eq::assert_approx_eq;
    use ndarray::{array, s, Array2};
    use ndarray_linalg::assert_close_l1;

    use crate::{fiducial_detector::FidDetectionParameter, utils::ImageUtil};

    use super::{
        fiducial_registration, homography, refine_individual, registration_error,
        similarity_transform, transform_pts_2d, Coord, FiducialFrame, FiducialPoint,
    };

    const SRC: [Coord; 5] = [
        Coord(0., 0.),
        Coord(0., 0.5),
        Coord(0., 1.),
        Coord(1., 0.),
        Coord(1., 1.),
    ];
    const DEST: [Coord; 5] = [
        Coord(2.0, 0.5),
        Coord(2.0, 1.5),
        Coord(2.0, 2.5),
        Coord(4.0, 1.0),
        Coord(4.0, 2.0),
    ];

    #[test]
    fn test_2d_transform() {
        let transform = homography(&SRC, &DEST).unwrap();
        let transformed_points = transform_pts_2d(&SRC, &transform);
        let total_error: f64 = std::iter::zip(&transformed_points, &DEST)
            .map(|(x, y)| ((x.0 - y.0).powi(2) + (x.1 - y.1).powi(2)).sqrt())
            .sum();
        assert!(total_error < 0.001);
    }

    #[test]
    fn test_registration_error() {
        let a = vec![Coord(0., 0.), Coord(1., 0.)];
        let b = vec![Coord(0., 1.), Coord(0., 0.)];
        assert_eq!(registration_error(&a, &b, &Array2::eye(3)), vec![1., 1.]);
    }

    #[test]
    fn test_similarity() {
        let src = Coord::from_arr(&[[0., 0.], [0., 0.5], [0., 1.], [1., 0.], [1., 1.]]);
        let target = Coord::from_arr(&[
            [2., 1.],
            [1.1339746, 1.5],
            [0.26794919, 2.],
            [3., 2.73205081],
            [1.26794919, 3.73205081],
        ]);
        let expected = array![[1., -1.73205081, 2.], [1.73205081, 1., 1.], [0., 0., 1.]];
        let actual = similarity_transform(&src, &target);
        expected
            .iter()
            .zip(actual.iter())
            .for_each(|(x, y)| assert_approx_eq!(x, y));
    }

    #[test]
    fn test_homography() {
        let expected: Array2<f64> = array![[6.0, 0.0, 2.0], [1.5, 2.0, 0.5], [1.0, 0.0, 1.0]];
        let actual = homography(&SRC, &DEST).unwrap();
        assert_close_l1!(&expected, &actual, 0.1);
    }
    #[test]
    fn test_fiducial_registration() {
        let file = File::open("testing_data/design.txt").expect("could not open design file");
        let buf = BufReader::new(file);
        let temp: Vec<FiducialPoint> = buf.lines().map(|x| x.unwrap().parse().unwrap()).collect();
        let mut design = FiducialFrame::new(temp);

        let file = File::open("testing_data/detect.txt").expect("could not open design file");
        let buf = BufReader::new(file);
        let temp: Vec<FiducialPoint> = buf.lines().map(|x| x.unwrap().parse().unwrap()).collect();
        let mut detect = FiducialFrame::new(temp);

        let (_homography, correct) = fiducial_registration(&mut detect, &mut design).unwrap();
        assert!(correct == 58, "homography does not cover all 58 fiducials")
    }

    #[test]
    fn test_refine_individual() {
        let image: Array2<f32> = ImageUtil::read_r_channel("testing_data/single_fiducial.tif");
        //pad image because this is not a full Cytassist image
        let mut full_image: Array2<f32> =
            Array2::zeros((image.shape()[0] + 100, image.shape()[1] + 100));
        full_image
            .slice_mut(s![50..image.shape()[0] + 50, 50..image.shape()[1] + 50])
            .assign(&image);
        let (x, y): (usize, usize) = (90, 90);
        let bbox_width = 50;
        let image = full_image
            .slice(s![
                x - bbox_width..x + bbox_width,
                y - bbox_width..y + bbox_width
            ])
            .to_owned();

        //uncomment to check image slice
        /*
        let gray = ImageUtil::to_gray_image(&image);
        gray.save("single_fiducial.tif")
            .expect("could not write out single fiducial image");
        */
        let param = FidDetectionParameter::new_visium_hd_param();
        let center =
            refine_individual(&image, &param, bbox_width as f64).expect("unable to find fiducial");

        let center = center.sum(&Coord((y - 50) as f64, (x - 50) as f64));
        let actual_center = Coord(42.722523, 43.691833);
        assert!(
            (center.0 - actual_center.0).abs() < 0.01 && (center.1 - actual_center.1).abs() < 0.01,
            "Center refinement incorrect"
        );
    }
}
