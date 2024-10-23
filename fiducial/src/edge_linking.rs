use ndarray::{s, Array2};
use ndarray_npy::write_npy;

use crate::{
    bresenham::bresenham, edge_subpixel::write_vec, fiducial_detector::FidDetectionParameter,
};

#[allow(dead_code)]
pub(crate) struct Links {
    pub link_count: usize,
    pub link_list: Array2<usize>,
    pub link_centers: Vec<[f32; 2]>,
    pub link_directions: Vec<[f32; 2]>,
}

pub(crate) struct EdgeData {
    pub pts: Vec<[f32; 2]>,      // (edge_x, edge_y)
    pub pixels: Vec<[usize; 2]>, // (edge_row, edge_col)
    pub norms: Vec<[f32; 2]>,    // (edge_norm_x, edge_norm_y)
    pub rows: usize,
    pub cols: usize,
}

impl Links {
    #[allow(dead_code)]
    pub(crate) fn write(&self, path: &str) {
        let ll32 = self.link_list.map(|x| *x as u32);
        write_npy(format!("{}/link_list.npy", path), &ll32).unwrap();
        write_vec(&format!("{}/link_centers.txt", path), &self.link_centers);
        write_vec(
            &format!("{}/link_directions.txt", path),
            &self.link_directions,
        );
    }
}

pub(crate) fn edge_linking(edge_data: &EdgeData, fid: &FidDetectionParameter) -> Links {
    let (n_rows, n_cols) = (edge_data.rows, edge_data.cols);
    let mut vote_map_index: Array2<i32> = Array2::from_elem((n_rows, n_cols), -1); //vote_map[0]
    for (edge_id, &pt) in edge_data.pixels.iter().enumerate() {
        vote_map_index[(pt[0], pt[1])] = edge_id as i32;
    }
    _edge_linking(edge_data, fid, vote_map_index)
}

pub(crate) fn _edge_linking(
    edge_data: &EdgeData,
    fid: &FidDetectionParameter,
    vote_map_index: Array2<i32>,
) -> Links {
    let n_edges = edge_data.pixels.len();
    let (n_rows, n_cols) = (edge_data.rows, edge_data.cols);
    let mut vote_map_count: Array2<usize> = Array2::zeros((n_rows, n_cols)); //vote_map[1]
    let mut link_list: Array2<_> = Array2::zeros((n_edges, fid.link_length));
    let mut link_centers = Vec::new();
    let mut link_directions = Vec::new();

    //temp arrays
    let mut current_link = vec![0; fid.link_length];
    let mut search_line = vec![[0; 2]; 3 * fid.search_end + 1];
    let mut link_count = 0;

    for i in 0..n_edges {
        if vote_map_count[edge_data.pixels[i]] > 0 {
            continue;
        }
        let mut current_idx = i;
        let mut max_level = 0;
        let mut too_many_points = false;
        for (level, current_link_level) in current_link.iter_mut().enumerate() {
            max_level = level;
            let mut found_next = false;
            too_many_points = false;
            let x = edge_data.pts[current_idx][0];
            let y = edge_data.pts[current_idx][1];
            let n_x = edge_data.norms[current_idx][0];
            let n_y = edge_data.norms[current_idx][1];
            let direction = f32::powi(-1.0, (level % 2) as i32 + 1);
            let x_start = fid.search_start as f32 * direction * n_x + x;
            let y_start = fid.search_start as f32 * direction * n_y + y;
            let x_end = fid.search_end as f32 * direction * n_x + x;
            let y_end = fid.search_end as f32 * direction * n_y + y;
            let mut next_idx = 0;
            let n_points = bresenham(
                &mut search_line,
                fid.search_end as i32,
                (x_start.round() as i32, y_start.round() as i32),
                (x_end.round() as i32, y_end.round() as i32),
                true,
            );

            for [next_c, next_r] in search_line.iter().take(n_points).copied() {
                if (next_c < 0)
                    || (next_r < 0)
                    || (next_r >= n_rows as i32 - 1)
                    || (next_c >= n_cols as i32 - 1)
                {
                    break;
                }
                if vote_map_index[[next_r as usize, next_c as usize]] >= 0 {
                    next_idx = vote_map_index[[next_r as usize, next_c as usize]] as usize;
                    let angle =
                        n_x * edge_data.norms[next_idx][0] + n_y * edge_data.norms[next_idx][1];
                    if angle < fid.parallel_threshold {
                        found_next = true;
                        break;
                    }
                }
            }

            if found_next && level == fid.link_length - 1 {
                too_many_points = true;
                break;
            } else if found_next || level == fid.link_length - 1 {
                *current_link_level = current_idx;
                current_idx = next_idx;
            } else {
                break;
            }
        }

        if (max_level == fid.link_length - 1) && !too_many_points {
            let (mut center_x, mut center_y) = (0.0, 0.0);
            for l in 0..fid.link_length {
                let next_edge_idx = current_link[l];
                link_list[[link_count, l]] = next_edge_idx;
                vote_map_count[edge_data.pixels[next_edge_idx]] += 1;
                center_x += edge_data.pts[next_edge_idx][0];
                center_y += edge_data.pts[next_edge_idx][1];
            }
            let direction = [
                edge_data.pts[link_list[[link_count, max_level]]][0]
                    - edge_data.pts[link_list[[link_count, 0]]][0],
                edge_data.pts[link_list[[link_count, max_level]]][1]
                    - edge_data.pts[link_list[[link_count, 0]]][1],
            ];
            link_directions.push(direction);
            link_centers.push([
                center_x / fid.link_length as f32,
                center_y / fid.link_length as f32,
            ]);

            link_count += 1;
        }
    }

    let link_list = link_list.slice(s![..link_count, ..]).to_owned();
    Links {
        link_count,
        link_list,
        link_centers,
        link_directions,
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::{
        edge_linking::edge_linking, edge_subpixel::compute_edge_data,
        fiducial_detector::FidDetectionParameter, utils::ImageUtil,
    };

    #[test]
    fn test_edge_linking() {
        let param = FidDetectionParameter::new_visium_hd_param();
        let image: Array2<f32> = ImageUtil::read_r_channel("testing_data/single_fiducial.tif");
        let edge_data = compute_edge_data(&image, &param);
        let links = edge_linking(&edge_data, &param);

        //Should get a decent number of links
        assert!(
            links.link_count > 100,
            "Only found {} edge links in the image, should get at least 100",
            links.link_count
        );

        //Should all be roughly equidistant from the actual center
        let center = [42.722523, 43.691833];
        let distances: Vec<_> = links
            .link_centers
            .iter()
            .map(|[x, y]| ((*x - center[0]).powi(2) + (*y - center[1]).powi(2)).sqrt())
            .collect();
        let mean_distance = distances.iter().fold(0.0, |a, b| a + *b) / distances.len() as f32;
        let max_distance = distances
            .iter()
            .map(|x| (*x - mean_distance).abs())
            .fold(0.0, f32::max);
        assert!(
            max_distance < 1.0,
            "Variance in implied radius between links too high"
        );
    }
}
