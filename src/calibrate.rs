use std::error::Error;

use opencv::{
    core::{Size}, 
};
use nalgebra as na;

pub fn generate_world_points(square_length: f64, pattern: Size) -> Result<Vec<na::Point2<f64>>, Box<dyn Error>> {
    let Size{width, height} = pattern;
    let mut world_points = Vec::new();
    for x in 0..width {
        for y in 0..height {
            world_points.push(na::Point2::<f64>::new(x as f64 * square_length, y as f64 * square_length));
        }
    }
    Ok(world_points)
}

pub fn computeH(img_points: &Vec<na::Point2<f64>>, world_points: &Vec<na::Point2<f64>>) -> Result<na::Matrix3::<f64>, Box<dyn Error>> {
    let num_points = img_points.len();
    assert_eq!(num_points, world_points.len());

    // at least 4 point if want to compute H
    assert!(num_points > 3);

    type MatrixXx9<T> = na::Matrix<T, na::Dyn, na::U9, na::VecStorage<T, na::Dyn, na::U9>>;
    type RowVector9<T> = na::Matrix<T, na::U1, na::U9, na::ArrayStorage<T, 1, 9>>;

    let mut A = MatrixXx9::<f64>::zeros(num_points * 2);

    let img_world_points_iter = img_points.iter().zip(world_points.iter());
    for (idx, (img_point, world_point)) in img_world_points_iter.enumerate() {
        let u = img_point.x;
        let v = img_point.y;
        let x_w = world_point.x;
        let y_w = world_point.y;
        let ax = RowVector9::<f64>::from_vec(vec![
            -x_w, -y_w, -1.0, 0.0, 0.0, 0.0, u*x_w, u*y_w, u 
        ]);
        let ay = RowVector9::<f64>::from_vec(vec![
            0.0, 0.0, 0.0, -x_w, -y_w, -1.0, v*x_w, v*y_w, v
        ]);
        A.set_row(2 * idx, &ax);
        A.set_row(2 * idx + 1, &ay);
    } 
    let svd = A.svd(false, true);
    let v_t = match svd.v_t {
        Some(v_t) => v_t,
        None => return Err(From::from("compute V failed")),
    };
    let last_row = v_t.row(v_t.nrows() - 1);

    println!("{}", last_row);
    
    Ok(na::Matrix3::<f64>::from_iterator(last_row.into_iter().cloned()).transpose())
}

#[cfg(test)]
mod test {

use std::error::Error;

use opencv::{
    prelude::*,
    imgcodecs,
    imgproc,
    core::{self, Size}, 
};
use nalgebra as na;

#[test]
fn test_generate_world_points() -> Result<(), Box<dyn Error>> {
    let pattern = Size::new(7, 5);
    let points = crate::calibrate::generate_world_points(30.5, pattern)?;
    for p in points {
        println!("{}", p);
    }

    Err(From::from("end"))
}

#[test]
fn test_nalgebra() -> Result<(), Box<dyn Error>> {
    let v = na::RowVector4::<f64>::from_vec(vec![1., 2., 3., 4.]);
    let m = na::Matrix2::<f64>::from_column_slice(v.as_slice());
   
    print!("{}", v);
    print!("{}", m);


    Err(From::from("end"))
}

#[test]
fn test_computeH() -> Result<(), Box<dyn Error>> {
    let image_path = "./cali.jpg"; // Replace with your image path
    // Load the image in color mode
    let mut image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    let mut gray = Mat::default();
    imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Check if the image is empty
    if gray.empty() {
        println!("Failed to open the image.");
        return Err(From::from("gray image is empty"));
    }
    let pattern = Size::new(7, 5);
    let world_points = crate::calibrate::generate_world_points(30.5, pattern)?;
    let img_points = crate::detect::detect_corners(&gray)?;
    let h = super::computeH(&img_points, &world_points)?;
    println!("{}", h);

    // project frame to image
    let o_w = na::Point3::<f64>::new(0.0, 0., 1.);
    let x_w = na::Point3::<f64>::new(100.0, 0., 1.);
    let y_w = na::Point3::<f64>::new(0.0, 100., 1.);

    let o_i = h * o_w;
    let x_i = h * x_w;
    let y_i = h * y_w;
    println!("{},{},{}", o_i, x_i, y_i);

    imgproc::line(&mut image,
        core::Point2i::new(o_i.x as i32, o_i.y as i32),
        core::Point2i::new(x_i.x as i32, x_i.y as i32),
        core::Scalar::new(255.0, 0.0, 0.0, 255.0),
        5,
        imgproc::LINE_8,
        0)?;
    imgproc::line(&mut image,
        core::Point2i::new(o_i.x as i32, o_i.y as i32),
        core::Point2i::new(y_i.x as i32, y_i.y as i32),
        core::Scalar::new(255.0, 0.0, 0.0, 255.0),
        50,
        imgproc::LINE_8,
        0)?;

    crate::gui::imshow("title", &image)?;

    Err(From::from("end"))
}

}