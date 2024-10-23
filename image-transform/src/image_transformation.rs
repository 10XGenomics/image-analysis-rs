use image::Rgb;
use imageproc::geometric_transformations::{warp, Interpolation, Projection};
use ndarray::{arr1, arr2, Array2};
use rand::Rng;
use std::convert::TryInto;

fn create_temporary_tiff_file_path() -> Result<String, String> {
    let temporary_output_dir = std::env::temp_dir();
    let mut temporary_file_name: String = rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(10)
        .map(char::from)
        .collect();
    temporary_file_name.push_str(".tiff");
    let temporary_path = temporary_output_dir.as_path().join(temporary_file_name);

    // Possibility that Windows utf-16 paths contain data that cannot be
    // re-encoded to utf-8:
    // https://github.com/rust-lang/rust/issues/12056
    let path_string = temporary_path
        .clone()
        .into_os_string()
        .into_string()
        .map_err(|_| {
            format!(
                "Failed to convert path to string {}",
                temporary_path.display()
            )
        })?;

    Ok(path_string)
}

pub fn image_perspective_transform(
    image_path: &str,
    optional_output_path: Option<String>,
    perspective_transform: [f32; 9],
) -> Result<String, String> {
    /* Applies an isolated tilt matrix to the image from the same image path.
       Outputs absolute path to transformed image on disk if successful.
       This implementation should mimic what the pipeline is doing (but with OpenCV)
       as closely as possible.

       Args:
          image_path: Path to the image
          perspective_transform: isolated tilt transform. The domain and range
            of this transform use corner-based image coordinates
          optional_output_path: A path to write the transformed image to. If no value is provided
            it will be written to a temporary directory as provided by the OS's APIs.
    */
    let image = image::open(image_path).unwrap().into_rgb8();

    // Convert the corner-based coordinate transform into a 2d array, then left- and right-multiply
    // to create `corrected_transform` which works with center-based coordinates
    let transform_mat: Array2<_> = arr1(&perspective_transform).into_shape((3, 3)).unwrap();
    let center_to_corner = arr2(&[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]);
    let corner_to_center = arr2(&[[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]]);
    let corrected_transform_mat = &corner_to_center.dot(&transform_mat).dot(&center_to_corner);
    let corrected_transform: [f32; 9] = corrected_transform_mat
        .as_slice()
        .expect("array not contiguous")
        .try_into()
        .expect("transform matrix not size 3x3");
    let projection = Projection::from_matrix(corrected_transform).unwrap();
    // OpenCV uses bilinear by default for their warpPerspective.
    // https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    // And the default value is also being used in the pipeline.
    let interpolation = Interpolation::Bilinear;
    let default_pixel_color = Rgb([255u8, 255u8, 255u8]);
    let warped_image = warp(&image, &projection, interpolation, default_pixel_color);
    let output_path = match optional_output_path {
        Some(result) => result,
        None => create_temporary_tiff_file_path()?,
    };

    warped_image.save(&output_path).map_err(|e| e.to_string())?;

    Ok(output_path)
}

#[cfg(test)]
mod tests {
    use crate::image_transformation::{
        create_temporary_tiff_file_path, image_perspective_transform,
    };
    use image::RgbImage;

    #[test]
    fn test_image_perspective_transform() {
        let path = match create_temporary_tiff_file_path() {
            Ok(result) => result,
            Err(err) => panic!("{}", err),
        };
        let image = RgbImage::new(32, 32);
        let _ = image.save(path.clone());

        // TODO
        // Identity matrix since we can't unit test that the transform happened the
        // way we expected. We could possibly do so by reading pixels and confirming
        // that one pixel moved to a new place or something like that.
        let transform = [
            1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32,
        ];

        let transformed_image_path =
            image_perspective_transform(&path, Some(path.clone()), transform).unwrap();
        assert!(path == transformed_image_path)
    }
}
