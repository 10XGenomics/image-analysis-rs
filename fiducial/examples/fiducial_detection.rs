//Perform Fiducial Detection and Decoding on an input image

use clap::Parser;
use fiducial::{
    fiducial_detector::{detect_fiducial, FidDetectionParameter},
    utils::ImageUtil,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    //path to RGB TIFF file
    #[arg(short, long)]
    image_path: String,
}

fn main() {
    //read input image
    let args = Args::parse();
    let full_image = ImageUtil::read_r_channel(&args.image_path);
    let param = FidDetectionParameter::new_visium_hd_param();
    let (detected_names, detected_centers): (Vec<_>, Vec<_>) =
        detect_fiducial(&full_image, &param, &None)
            .iter()
            .cloned()
            .unzip();
    detected_names
        .iter()
        .zip(detected_centers)
        .for_each(|(name, center)| println!("{}: {:?}", name, center));
}
