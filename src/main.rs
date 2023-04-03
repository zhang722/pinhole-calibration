use std::error::Error;

use opencv::{
    prelude::*,
    imgcodecs,
    imgproc,
    core::{self}, 
};

mod detect;
mod gui;
mod calibrate;

fn main() -> Result<(), Box<dyn Error>> {
    let image_path = "../cali.jpg"; // Replace with your image path
    // Load the image in color mode
    let mut image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    let mut gray = Mat::default();
    imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Check if the image is empty
    if gray.empty() {
        println!("Failed to open the image.");
        return Err(From::from("gray image is empty"));
    }

    let points = detect::detect_corners(&gray)?; 

    for (idx, p) in points.iter().enumerate() {
        imgproc::circle(&mut image, core::Point2i::new(p.x as i32, p.y as i32), 50, core::Scalar::new(255.0, 0.0, 0.0, 255.0), 5, 
            imgproc::LINE_8, 0)?;
        println!("{:?}", p);
    }

    gui::imshow("title", &image)?;

    Ok(())
}

