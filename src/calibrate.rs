use std::error::Error;

use opencv::{
    core::{Size, Point2d}, 
};
use nalgebra;

pub fn generate_world_points(square_length: f64, pattern: Size) -> Result<Vec<Point2d>, Box<dyn Error>> {
    let Size{width, height} = pattern;
    let mut world_points = Vec::new();
    for x in 0..width {
        for y in 0..height {
            world_points.push(Point2d::new(x as f64 * square_length, y as f64 * square_length));
        }
    }
    Ok(world_points)
}

// pub fn computeH(img_points: Vec<Point2f>, world_points: Vec<Point2f>) -> Result<>

#[cfg(test)]
mod test {

use std::error::Error;

use opencv::{
    core::{Size}, 
};
use nalgebra as na;

#[test]
fn test_generate_world_points() -> Result<(), Box<dyn Error>> {
    let pattern = Size::new(7, 5);
    let points = crate::calibrate::generate_world_points(30.5, pattern)?;
    for p in points {
        println!("{:?}", p);
    }

    Err(From::from("end"))
}

#[test]
fn test_nalgebra() -> Result<(), Box<dyn Error>> {
    let m: na::Matrix3<f64> = na::Matrix3::zeros();
    let m = m.insert_row(3, 1.0);

    println!("{}", m);

    Err(From::from("end"))
}

}