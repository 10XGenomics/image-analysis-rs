use crate::utils::{pt_angle, pt_dist};
use itertools::Itertools;
use ndarray::{arr1, Array2};
use ndarray_linalg::svd::SVD;

///Fit a circle with x, y coordinate.
pub fn circle_fitting(
    points: &[[f32; 2]],
    radius_up_limit: f32,
    radius_low_limit: f32,
    radius_var_limit: f32,
    angle_spanning_limit: f32,
) -> ([f32; 2], f32, bool) {
    let n_points = points.len();

    let mut a = Array2::zeros((n_points, 4));
    for i in 0..n_points {
        let (x, y) = (points[i][0] as f64, points[i][1] as f64);
        a[[i, 0]] = x;
        a[[i, 1]] = y;
        a[[i, 2]] = 1.0;
        a[[i, 3]] = x * x + y * y;
    }

    let v = a.svd(false, true).unwrap().2.unwrap();
    let center = [
        (-0.5 * v[(3, 0)] / v[(3, 3)]) as f32,
        (-0.5 * v[(3, 1)] / v[(3, 3)]) as f32,
    ];

    //TODO: This algorithm underestimates the actual span of the circle
    // (e.g. top right and bottom left quadrants of unit circle result in ~2.8 radians, not PI)
    let angle_list: Vec<_> = points.iter().map(|val| pt_angle(val, &center)).collect();
    let positive: Vec<_> = angle_list.iter().filter(|x| **x >= 0.0).copied().collect();
    let negative: Vec<_> = angle_list.iter().filter(|x| **x < 0.0).copied().collect();
    let posmax = positive.iter().copied().reduce(f32::max);
    let posmin = positive.into_iter().reduce(f32::min);
    let negmax = negative.iter().copied().reduce(f32::max);
    let negmin = negative.into_iter().reduce(f32::min);

    let mut angle_span = 0.0;
    if let (Some(i), Some(j)) = (posmax, posmin) {
        angle_span += i - j;
    };
    if let (Some(i), Some(j)) = (negmax, negmin) {
        angle_span += i - j;
    };

    let r_list = points.iter().map(|val| pt_dist(&center, val)).collect_vec();
    let radius = arr1(&r_list).mean().unwrap();
    let radius_var: f32 = arr1(&r_list).var(1.);

    let mut bad_circle_fit = false;
    if radius <= radius_low_limit
        || radius >= radius_up_limit
        || radius_var / radius > radius_var_limit
        || angle_span < angle_spanning_limit
    {
        bad_circle_fit = true;
    }
    (center, radius, bad_circle_fit)
}

#[cfg(test)]
mod tests {

    use std::f32::consts::PI;

    use crate::bresenham::Circle;
    use crate::circle_fitting::circle_fitting;
    use crate::utils::set_log_config;
    use assert_approx_eq::assert_approx_eq;

    use itertools::Itertools;

    use slog::info;

    #[test]
    fn test_circle_fit() {
        let pts = vec![
            [13.405106, 50.0],
            [13.235407, 51.0],
            [13.081232, 52.0],
            [13.005961, 53.0],
            [12.905699, 54.0],
            [12.810581, 55.0],
            [12.8005705, 56.0],
            [12.858491, 57.0],
            [12.845777, 58.0],
            [13.007911, 59.0],
            [13.031402, 60.0],
            [13.156626, 61.0],
            [13.293638, 62.0],
            [13.55237, 63.0],
            [13.721881, 64.0],
            [13.957147, 65.0],
            [14.216442, 66.0],
            [14.627488, 67.0],
            [14.805289, 68.0],
            [15.133758, 69.0],
            [15.700414, 70.0],
            [16.091585, 71.0],
            [16.54973, 72.0],
            [16.94201, 73.0],
            [17.566103, 74.0],
            [18.12598, 75.0],
            [18.907465, 76.0],
            [19.659838, 77.0],
            [20.212326, 78.0],
            [21.043053, 79.0],
            [22.034958, 80.0],
            [22.987303, 81.0],
            [73.6949, 81.0],
            [79.166, 39.0],
            [80.20983, 41.0],
            [81.57599, 44.0],
            [81.85378, 68.0],
            [77.061905, 77.0],
            [83.43336, 52.0],
            [83.7272, 55.0],
            [83.779015, 58.0],
            [83.5159, 60.0],
            [74.02271, 32.0],
            [75.00491, 33.0],
            [74.716255, 80.0],
            [75.847984, 34.0],
            [76.293236, 78.0],
            [75.62922, 79.0],
            [76.58753, 35.0],
            [77.35079, 36.0],
            [77.98479, 37.0],
            [78.33123, 75.0],
            [77.79424, 76.0],
            [78.713036, 38.0],
            [79.52572, 73.0],
            [78.96815, 74.0],
            [79.84447, 40.0],
            [80.022705, 72.0],
            [80.76315, 42.0],
            [81.17543, 43.0],
            [80.99218, 70.0],
            [80.59748, 71.0],
            [81.98693, 45.0],
            [82.21165, 46.0],
            [82.423294, 47.0],
            [82.38555, 66.0],
            [82.09942, 67.0],
            [82.786896, 48.0],
            [83.02463, 49.0],
            [83.16588, 50.0],
            [83.33895, 51.0],
            [83.453415, 61.0],
            [83.245384, 62.0],
            [83.159256, 63.0],
            [82.89129, 64.0],
            [82.745674, 65.0],
            [83.66661, 53.0],
            [83.63723, 54.0],
            [83.861984, 56.0],
            [83.70115, 57.0],
            [83.69448, 59.0],
            [52.0, 21.01344],
            [53.0, 21.08141],
            [54.0, 21.262695],
            [55.0, 21.445894],
            [56.0, 21.709587],
            [57.0, 21.91379],
            [58.0, 22.233452],
            [59.0, 22.496616],
            [60.0, 22.923754],
            [61.0, 23.155233],
            [62.0, 23.657448],
            [63.0, 24.081959],
            [64.0, 24.574156],
            [65.0, 25.071257],
            [66.0, 25.71577],
            [67.0, 26.167845],
            [68.0, 26.931564],
            [69.0, 27.380987],
            [70.0, 28.136332],
            [71.0, 29.000807],
            [72.0, 29.9904],
            [73.0, 30.905382],
            [54.0, 21.262695],
            [28.0, 85.197266],
            [31.0, 87.15356],
            [70.0, 84.253204],
            [32.0, 87.69333],
            [38.0, 90.16559],
            [60.0, 89.65807],
            [60.0, 89.65807],
            [65.0, 86.86673],
            [41.0, 90.869934],
            [47.0, 91.66709],
            [49.0, 91.77064],
            [24.0, 82.01981],
            [73.0, 81.84817],
            [25.0, 82.946365],
            [71.0, 83.5171],
            [72.0, 82.69612],
            [26.0, 83.83383],
            [27.0, 84.74348],
            [69.0, 85.040436],
            [29.0, 85.98676],
            [30.0, 86.51051],
            [67.0, 86.10556],
            [68.0, 85.59688],
            [66.0, 86.57578],
            [33.0, 88.177956],
            [34.0, 88.56524],
            [63.0, 88.551384],
            [64.0, 87.82822],
            [35.0, 89.08379],
            [36.0, 89.445625],
            [61.0, 89.39755],
            [62.0, 88.99794],
            [37.0, 89.92149],
            [39.0, 90.432625],
            [57.0, 90.41521],
            [58.0, 90.19886],
            [59.0, 90.01663],
            [40.0, 90.74748],
            [42.0, 91.24629],
            [43.0, 91.292015],
            [44.0, 91.43417],
            [45.0, 91.4705],
            [50.0, 91.58234],
            [51.0, 91.53071],
            [52.0, 91.416985],
            [53.0, 91.27664],
            [54.0, 91.22989],
            [55.0, 91.02538],
            [56.0, 90.71134],
            [48.0, 91.72335],
        ];

        let (center, radius, bad_fit) = circle_fitting(&pts, 100.0, 4.0, 0.1, PI);
        assert!(
            (center[0] - 48.2922).abs() < 0.01 && (center[1] - 56.246983).abs() < 0.01,
            "Center incorrect"
        );
        assert!((radius - 35.4318).abs() < 0.01, "Radius incorrect");
        assert!(!bad_fit, "Circle fit should not be bad in this case")
    }

    #[test]
    fn test_circle_fit_angle_rejection() {
        //100 points along 1/4 of circle, should be rejected
        let mut pts: Vec<_> = (0..100)
            .map(|i| {
                let x = i as f32 / 100.0;
                let y = (1.0 - x * x).sqrt();
                [x, y]
            })
            .collect();
        let circle = circle_fitting(&pts, 2.0, 0.0, 0.1, PI);
        assert!(circle.2);

        let bottom_left = (0..100).map(|i| {
            let x = -1.0 * i as f32 / 100.0;
            let y = -1.0 * (1.0 - x * x).sqrt();
            [x, y]
        });

        //TODO: This extra set of points should not be needed to pass the angle threshold
        let top_left = (0..100).map(|i| {
            let x = -1.0 * i as f32 / 100.0;
            let y = (1.0 - x * x).sqrt();
            [x, y]
        });
        pts.extend(bottom_left);
        pts.extend(top_left);
        let circle = circle_fitting(&pts, 2.0, 0.0, 0.1, PI);
        assert!(!circle.2);
    }

    #[test]
    fn test_circle_fitting() {
        let log = set_log_config();
        let center = (200., 200.);
        let radius = 30.;
        let radius_up_limit = 50.0;
        let radius_low_limit = 10.0;
        let radius_var_limit = 0.2;

        let points = Circle::new(center.0, center.1, radius);
        let points = points.map(|(x, y)| [x, y]).collect_vec();

        let (center_fit, radius_fit, bad_circle_fit) = circle_fitting(
            &points,
            radius_up_limit,
            radius_low_limit,
            radius_var_limit,
            PI,
        );

        info!(
            log,
            "center {:?}     radius {}   bad_circle_file {}",
            center_fit,
            radius_fit,
            bad_circle_fit
        );
        assert_approx_eq!(center_fit[0] as f64, center.0 as f64, 1e-1f64);
        assert_approx_eq!(center_fit[1] as f64, center.1 as f64, 1e-1f64);
        assert_approx_eq!(radius_fit as f64, radius as f64, 1e-2f64);
    }
}
