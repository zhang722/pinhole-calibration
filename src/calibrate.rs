use std::error::Error;

use opencv::{
    core::{Size}, imgcodecs, prelude::{Mat, MatTraitConst}, imgproc, 
};
use nalgebra as na;

fn Normalize(point_vec: &Vec<na::Point2<f64>>) 
-> Result<(Vec<na::Point2<f64>>, na::Matrix3<f64>), Box<dyn Error>>
{
    let mut norm_T = na::Matrix3::<f64>::identity();
    let mut normed_point_vec = Vec::new();
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for p in point_vec {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= point_vec.len() as f64;
    mean_y /= point_vec.len() as f64;
    let mut mean_dev_x = 0.0;
    let mut mean_dev_y = 0.0;
    for p in point_vec {
        mean_dev_x += (p.x - mean_x).abs();
        mean_dev_y += (p.y - mean_y).abs();
    }
    mean_dev_x /= point_vec.len() as f64;
    mean_dev_y /= point_vec.len() as f64;
    let sx = 1.0 / mean_dev_x;
    let sy = 1.0 / mean_dev_y;

    for p in point_vec {
        let mut p_tmp = na::Point2::<f64>::new(0.0, 0.0);
        p_tmp.x = sx * p.x - mean_x * sx;
        p_tmp.y = sy * p.y - mean_y * sy;
        normed_point_vec.push(p_tmp);
    }
    norm_T[(0, 0)] = sx;
    norm_T[(0, 2)] = -mean_x * sx;
    norm_T[(1, 1)] = sy;
    norm_T[(1, 2)] = -mean_y * sy;

    Ok((normed_point_vec, norm_T))
}

pub fn generate_world_points(square_length: f64, pattern: Size) -> Result<Vec<na::Point2<f64>>, Box<dyn Error>> {
    let Size{width, height} = pattern;
    let mut world_points = Vec::new();
    for x in 0..height {
        for y in 0..width {
            world_points.push(na::Point2::<f64>::new(y as f64 * square_length, x as f64 * square_length));
        }
    }
    Ok(world_points)
}

pub fn compute_h(img_points: &Vec<na::Point2<f64>>, world_points: &Vec<na::Point2<f64>>) -> Result<na::Matrix3::<f64>, Box<dyn Error>> {
    let num_points = img_points.len();
    assert_eq!(num_points, world_points.len());

    // at least 4 point if want to compute H
    assert!(num_points > 3);

    type MatrixXx9<T> = na::Matrix<T, na::Dyn, na::U9, na::VecStorage<T, na::Dyn, na::U9>>;
    type RowVector9<T> = na::Matrix<T, na::U1, na::U9, na::ArrayStorage<T, 1, 9>>;

    let norm_img = Normalize(img_points)?;
    let norm_world = Normalize(world_points)?;

    let mut a = MatrixXx9::<f64>::zeros(num_points * 2);

    let img_world_points_iter = norm_img.0.iter().zip(norm_world.0.iter());
    for (idx, (img_point, world_point)) in img_world_points_iter.enumerate() {
        let u = img_point.x;
        let v = img_point.y;
        let x_w = world_point.x;
        let y_w = world_point.y;

        let ax = RowVector9::<f64>::from_vec(vec![
            x_w, y_w, 1.0, 0.0, 0.0, 0.0, -u*x_w, -u*y_w, -u 
        ]);
        let ay = RowVector9::<f64>::from_vec(vec![
            0.0, 0.0, 0.0, x_w, y_w, 1.0, -v*x_w, -v*y_w, -v
        ]);
        
        a.set_row(2 * idx, &ax);
        a.set_row(2 * idx + 1, &ay);
    } 
    let svd = a.svd(true, true);
    let v_t = match svd.v_t {
        Some(v_t) => v_t,
        None => return Err(From::from("compute V failed")),
    };
    let last_row = v_t.row(8);

    // normalize
    // let last_row = last_row / last_row[8];

    // construct matrix from vector
    let mut ret = na::Matrix3::<f64>::from_iterator(last_row.into_iter().cloned()).transpose();


    ret = match norm_img.1.try_inverse() {
        Some(m) => m,
        None => return Err(From::from("compute inverse norm_img failed")),
    } * ret * norm_world.1;
    
    Ok(ret)  
}


pub fn compute_b(homos: &Vec<na::Matrix3<f64>>) -> Result<na::Matrix3<f64>, Box<dyn Error>> {
    let num_homos = homos.len();
    // at least 3 homography matrices
    assert!(num_homos > 2);

    type MatrixXx6<T> = na::Matrix<T, na::Dyn, na::U6, na::VecStorage<T, na::Dyn, na::U6>>;
    type RowVector6<T> = na::Matrix<T, na::U1, na::U6, na::ArrayStorage<T, 1, 6>>;

    let get_v = |h: &na::Matrix3<f64>, i: usize, j: usize| -> RowVector6<f64> {
        let h1 = h.column(0);
        let h2 = h.column(1);
        let h3 = h.column(2);
        let i = i - 1;
        let j = j - 1;
        RowVector6::<f64>::from_vec(vec![
            h1[i] * h1[j], h1[i] * h2[j] + h2[i] * h1[j], h3[i] * h1[j] + h1[i] * h3[j],
            h2[i] * h2[j], h3[i] * h2[j] + h2[i] * h3[j], h3[i] * h3[j],
        ])
    };        

    let mut a = MatrixXx6::<f64>::zeros(num_homos * 2);

    for (idx, h) in homos.iter().enumerate() {
        // 第1列
        let h11 = h[(0, 0)];
        let h21 = h[(1, 0)];
        let h31 = h[(2, 0)];
        // 第2列
        let h12 = h[(0, 1)];
        let h22 = h[(1, 1)];
        let h32 = h[(2, 1)];

        let v11 = RowVector6::<f64>::new(h11 * h11, h11 * h21 + h11 * h21, h21 * h21, h11 * h31 + h31 * h11, h21 * h31 + h31 * h21 + h31 * h31, h31 * h31);
        let v12 = RowVector6::<f64>::new(h11 * h12, h11 * h22 + h21 * h12, h21 * h22, h11 * h32 + h31 * h12, h21 * h32 + h31 * h22 + h31 * h32, h31 * h32);
        let v22 = RowVector6::<f64>::new(h12 * h12, h12 * h22 + h12 * h22, h22 * h22, h12 * h32 + h32 * h12, h22 * h32 + h32 * h22 + h32 * h32, h32 * h32);

        a.set_row(2 * idx, &v12);
        a.set_row(2 * idx + 1, &(v11 - v22));
    } 
    let svd = a.svd(false, true);
    let v_t = match svd.v_t {
        Some(v_t) => v_t,
        None => return Err(From::from("compute V failed")),
    };
    let b = v_t.row(5);

    let b11 = b[(0, 0)];
    let b12 = b[(0, 1)];
    let b22 = b[(0, 2)];
    let b13 = b[(0, 3)];
    let b23 = b[(0, 4)];
    let b33 = b[(0, 5)];

    // construct matrix from vector
    let ret = na::Matrix3::<f64>::from_vec(vec![
        b11, b12, b13, b12, b22, b23, b13, b23, b33      
    ]);
    
    Ok(ret)  
}

pub fn compute_k(b: &na::Matrix3<f64>) -> Result<na::Matrix3<f64>, Box<dyn Error>> {
    let b11 = b[(0, 0)];
    let b12 = b[(0, 1)];
    let b13 = b[(0, 2)];
    let b22 = b[(1, 1)];
    let b23 = b[(1, 2)];
    let b33 = b[(2, 2)];
    let v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 * b12);
    let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    let alpha = (lambda / b11).sqrt();
    let beta = (lambda * b11 / (b11 * b22 - b12 * b12)).sqrt();
    let gamma = -b12 * alpha * alpha * beta / lambda;
    let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

    Ok(na::Matrix3::<f64>::from_vec(vec![
        alpha, 0., 0., gamma, beta, 0., u0, v0, 1., 
    ]))
}


pub fn compute_tf(h: &na::Matrix3<f64>, k: &na::Matrix3<f64>) -> Result<na::Isometry3<f64>, Box<dyn Error>>{
    let a = match k.try_inverse() {
        Some(m) => m, 
        None => return Err(From::from("k is not invertible")),
    } * h; 
    let r1 = a.column(0);
    let r2 = a.column(1);
    let r3 = r1.cross(&r2);
    let mut r = na::Matrix3::<f64>::zeros();
    r.set_column(0, &r1);
    r.set_column(1, &r2);
    r.set_column(2, &r3);
    
    let r = r.normalize();

    let r = na::Rotation3::<f64>::from_matrix_eps(&r, 1.0e-9, 10, na::Rotation3::identity());
    // let r = na::Rotation3::<f64>::from_matrix_unchecked(r);
    let t = a.column(2);
    let tf = 
        na::Isometry3::<f64>::from_parts(na::Translation3::new(t[0], t[1], t[2]), na::UnitQuaternion::from_rotation_matrix(&r));

    Ok(tf)
}


pub fn calibrate(size: (i32, i32)/*(width, height)*/) 
-> Result<(na::Matrix3<f64>, Vec<na::Isometry3<f64>>, Vec<Vec<na::Point2<f64>>>, Vec<na::Point2<f64>>), Box<dyn Error>> {
    let paths: Vec<String> = (0..=40).into_iter().map(|x| format!("./cali/{}.png", 100000 + x)).collect();
    let pattern = Size::new(size.0, size.1);
    let world_points = crate::calibrate::generate_world_points(0.02, pattern)?;
    let mut hs = Vec::new();
    let mut img_points_set = Vec::new();

    for path in &paths {
        let mut image = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Check if the image is empty
        if gray.empty() {
            println!("Failed to open the image.");
            return Err(From::from("gray image is empty"));
        }

        let img_points = crate::detect::detect_corners(&gray)?;
        if img_points.len() != (pattern.width * pattern.height) as usize {
            continue;
        }

        match compute_h(&img_points, &world_points) {
            Ok(h) => {
                hs.push(h);
                img_points_set.push(img_points);
            },
            Err(_) => continue,
        };
    }

    let b = compute_b(&hs)?;
    println!("b:{}", b);
    let k = compute_k(&b)?;
    println!("k:{}", k);

    let tfs: Vec<na::Isometry3<f64>> = hs.iter().map(|h| compute_tf(h, &k).expect("compute tf failed")).collect();

    Ok((k, 
        tfs, 
        img_points_set, 
        world_points))
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
    print!("{}", v);
    let m = na::Matrix2::<f64>::from_column_slice(v.as_slice());
    print!("{}", m);
    let m = na::Matrix2::<f64>::from_row_slice(v.as_slice());
    print!("{}", m);
    let m1 = na::Matrix2x3::<i32>::new(
        1, 2, 3, 4, 5, 6 
    );
    print!("{}", m1);
    let m1 = na::Matrix2x3::<i32>::from_vec(vec![
        1, 2, 3, 4, 5, 6 ]
    );
    print!("{}", m1);


    Err(From::from("end"))
}

#[test]
fn test_compute_h() -> Result<(), Box<dyn Error>> {
    let image_path = "./cali/100000.png"; // Replace with your image path
    // Load the image in color mode
    let mut image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    let mut gray = Mat::default();
    imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Check if the image is empty
    if gray.empty() {
        println!("Failed to open the image.");
        return Err(From::from("gray image is empty"));
    }
    let pattern = Size::new(11, 8);
    let world_points = crate::calibrate::generate_world_points(30.5, pattern)?;
    let img_points = crate::detect::detect_corners(&gray)?;
    for (idx, p) in img_points.iter().enumerate() {
        if idx > 7 {break;}
        imgproc::circle(&mut image, (p.x as i32, p.y as i32).into(), 10, core::Scalar::new(255.0, 0.0, 0.0, 255.0), 2, imgproc::LINE_8, 0)?;
    }
    let h = super::compute_h(&img_points, &world_points)?;

    println!("{}", h);

    // project frame to image
    let o_w = na::Point3::<f64>::new(0.0, 0., 1.);
    let x_w = na::Point3::<f64>::new(100.0, 0., 1.);
    let y_w = na::Point3::<f64>::new(0.0, 100., 1.);

    let o_i = h * o_w;
    let x_i = h * x_w;
    let y_i = h * y_w;
    let o_i = o_i / o_i[2];
    let x_i = x_i / x_i[2];
    let y_i = y_i / y_i[2];

    println!("{},{},{}", o_i, x_i, y_i);

    imgproc::line(&mut image,
        core::Point2i::new(o_i.x as i32, o_i.y as i32),
        core::Point2i::new(x_i.x as i32, x_i.y as i32),
        core::Scalar::new(255.0, 0.0, 0.0, 255.0),
        10,
        imgproc::LINE_8,
        0)?;
    imgproc::line(&mut image,
        core::Point2i::new(o_i.x as i32, o_i.y as i32),
        core::Point2i::new(y_i.x as i32, y_i.y as i32),
        core::Scalar::new(255.0, 0.0, 0.0, 255.0),
        10,
        imgproc::LINE_8,
        0)?;

    crate::gui::imshow("title", &image)?;

    Err(From::from("end"))
}

#[test]
fn test_compute_b() -> Result<(), Box<dyn Error>> {
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    let save_path = "./save.txt";
    let mut save_file = OpenOptions::new().append(true).open(save_path)?;


    let paths: Vec<String> = (0..=40).into_iter().map(|x| format!("./cali/{}.png", 100000 + x)).collect();
    let pattern = Size::new(11, 8);
    let world_points = crate::calibrate::generate_world_points(0.02, pattern)?;
    let mut hs = Vec::new();

    for path in &paths {
        let mut image = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Check if the image is empty
        if gray.empty() {
            println!("Failed to open the image.");
            return Err(From::from("gray image is empty"));
        }

        let img_points = crate::detect::detect_corners(&gray)?;
        if img_points.len() != (pattern.width * pattern.height) as usize {
            continue;
        }
        let h = super::compute_h(&img_points, &world_points)?;
        hs.push(h);
    }

    let b = super::compute_b(&hs)?;
    println!("b:{}", b);
    let k = super::compute_k(&b)?;
    println!("k:{}", k);

    Err(From::from("end"))
}


#[test]
fn test_cross() -> Result<(), Box<dyn Error>> {
    let cross = |u: na::Vector3<i32>, v: na::Vector3<i32>| -> na::Vector3<i32> {
        na::Vector3::<i32>::new(
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
    };

    let a = na::Vector3::new(1, 2, 3);
    let b = na::Vector3::new(4, 5, 6);
    println!("{}", a.cross(&b));
    println!("{}", cross(a, b));


    Err(From::from("end"))
}

}