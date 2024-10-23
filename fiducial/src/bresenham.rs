pub(crate) fn bresenham(
    line: &mut [[i32; 2]],
    search_r: i32,
    start_pt: (i32, i32),
    end_pt: (i32, i32),
    thick: bool,
) -> usize {
    let mut num_pts = 0;
    let (x0, y0) = start_pt;
    let (x1, y1) = end_pt;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let mut error_x = 2 * dx - dy;
    let mut error_y = 2 * dy - dx;
    let step_x = if x0 < x1 { 1 } else { -1 };
    let step_y = if y0 < y1 { 1 } else { -1 };

    let mut x = x0;
    let mut y = y0;
    for i in 0..(search_r + 1) {
        let mut x_moved = false;
        let mut y_moved = false;
        line[num_pts][0] = x;
        line[num_pts][1] = y;
        num_pts += 1;
        while error_x >= 0 {
            x_moved = true;
            x += step_x;
            error_x -= 2 * search_r;
        }
        error_x += 2 * dx;

        while error_y >= 0 {
            y_moved = true;
            y += step_y;
            error_y -= 2 * search_r;
        }
        error_y += 2 * dy;

        if thick && x_moved && y_moved && i < search_r {
            line[num_pts][0] = x - step_x;
            line[num_pts][1] = y;
            num_pts += 1;
            line[num_pts][0] = x;
            line[num_pts][1] = y - step_y;
            num_pts += 1;
        }
    }
    num_pts
}

pub struct Circle {
    x: f32,
    y: f32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    error: f32,
    quadrant: u8,
}

impl Circle {
    #[inline]
    pub fn new(center_x: f32, center_y: f32, radius: f32) -> Self {
        Self {
            center_x,
            center_y,
            radius,
            x: -radius,
            y: 0.0,
            error: 2. * (1.0 - radius),
            quadrant: 1,
        }
    }
}

impl Iterator for Circle {
    type Item = (f32, f32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.x < 0. {
            let point = match self.quadrant {
                1 => (self.center_x - self.x, self.center_y + self.y),
                2 => (self.center_x - self.y, self.center_y - self.x),
                3 => (self.center_x + self.x, self.center_y - self.y),
                4 => (self.center_x + self.y, self.center_y + self.x),
                _ => unreachable!(),
            };

            // Update the variables after each set of quadrants
            if self.quadrant == 4 {
                self.radius = self.error;

                if self.radius <= self.y {
                    self.y += 1.;
                    self.error += self.y * 2.0 + 1.0;
                }

                if self.radius > self.x || self.error > self.y {
                    self.x += 1.0;
                    self.error += self.x * 2.0 + 1.0;
                }
            }

            self.quadrant = self.quadrant % 4 + 1;

            Some(point)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{bresenham, Circle};
    use itertools::Itertools;

    #[test]
    fn test_circle() {
        let points = Circle::new(10., 40., 50.).collect_vec();
        let max_error = points
            .iter()
            .map(|(x, y)| ((*x - 10.0).powi(2) + (*y - 40.0).powi(2)).sqrt() - 50.0)
            .fold(0.0, f32::max);
        assert!(max_error <= 0.5, "Circle pixels off by more than 0.5")
    }

    #[test]
    fn test_bresenham() {
        let search_radius = 50;
        let mut search_line = vec![[0; 2]; search_radius + 1];
        let n_points = bresenham(
            &mut search_line,
            search_radius as i32,
            (10, 90),
            (10, -10),
            false,
        );

        let res = vec![
            [10i32, 90i32],
            [10, 87],
            [10, 85],
            [10, 83],
            [10, 81],
            [10, 79],
            [10, 77],
            [10, 75],
            [10, 73],
            [10, 71],
            [10, 69],
            [10, 67],
            [10, 65],
            [10, 63],
            [10, 61],
            [10, 59],
            [10, 57],
            [10, 55],
            [10, 53],
            [10, 51],
            [10, 49],
            [10, 47],
            [10, 45],
            [10, 43],
            [10, 41],
            [10, 39],
            [10, 37],
            [10, 35],
            [10, 33],
            [10, 31],
            [10, 29],
            [10, 27],
            [10, 25],
            [10, 23],
            [10, 21],
            [10, 19],
            [10, 17],
            [10, 15],
            [10, 13],
            [10, 11],
            [10, 9],
            [10, 7],
            [10, 5],
            [10, 3],
            [10, 1],
            [10, -1],
            [10, -3],
            [10, -5],
            [10, -7],
            [10, -9],
            [10, -11],
        ];
        assert_eq!(n_points, 51);
        assert_eq!(search_line, res);
    }
}
