//Perform fiducial detection and registration between an input image and
//the Visium HD design

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use clap::Parser;
use fiducial::utils::ImageUtil;
use fiducial::{
    fiducial_detector::{detect_fiducial, FidDetectionParameter},
    fiducial_registration::{
        fiducial_refinement_and_perspective_transform_isolation, fiducial_registration, Coord,
        FiducialFrame, FiducialPoint,
    },
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    //path to RGB TIFF file
    #[arg(short, long)]
    image_path: String,
}

fn main() {
    //read design
    let path = "testing_data/design.txt";
    let file = File::open(path).expect("could not read design file");
    let buf = BufReader::new(file);
    let temp: Vec<FiducialPoint> = buf.lines().map(|x| x.unwrap().parse().unwrap()).collect();
    let mut design = FiducialFrame::new(temp);

    //run fiducial detection
    let args = Args::parse();
    let full_image = ImageUtil::read_r_channel(&args.image_path);
    let param = FidDetectionParameter::new_visium_hd_param();
    let (detected_names, detected_centers): (Vec<_>, Vec<_>) =
        detect_fiducial(&full_image, &param, &None)
            .iter()
            .cloned()
            .unzip();

    let detected_centers: Vec<_> = detected_centers
        .iter()
        .map(|x| Coord(x[0] as f64, x[1] as f64))
        .collect();
    let mut detected_frame = FiducialFrame::from_points(&detected_names, &detected_centers);

    let homography = fiducial_registration(&mut detected_frame, &mut design)
        .expect("could not compute homography")
        .0;

    println!("Computed Homography:\n{:?}", homography);

    let refinement_with_perspective_transform =
        fiducial_refinement_and_perspective_transform_isolation(
            &args.image_path,
            &homography,
            &mut design,
        )
        .unwrap();
    println!(
        "Refinement with perspective transform isolation output:\n{:?}\n{:?}",
        refinement_with_perspective_transform.0, refinement_with_perspective_transform.1
    );
}
