use nalgebra as na;
use opencv::{imgproc, imgcodecs, core};
use crate::lm;

type Vector8<T> = na::Matrix<T, na::U8, na::U1, na::ArrayStorage<T, 8, 1>>;
type Matrix2x8<T> = na::Matrix<T, na::U2, na::U8, na::ArrayStorage<T, 2, 8>>;

/// Produces a skew-symmetric or "cross-product matrix" from
/// a 3-vector. This is needed for the `exp_map` and `log_map`
/// functions
fn skew_sym(v: na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ss = na::Matrix3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// Converts a 6-Vector Lie Algebra representation of a rigid body
/// transform to an NAlgebra Isometry (quaternion+translation pair)
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn exp_map(param_vector: &na::Vector6<f64>) -> na::Isometry3<f64> {
    let t = param_vector.fixed_view::<3, 1>(0, 0);
    let omega = param_vector.fixed_view::<3, 1>(3, 0);
    let theta = omega.norm();
    let half_theta = 0.5 * theta;
    let quat_axis = omega * half_theta.sin() / theta;
    let quat = if theta > 1e-6 {
        na::UnitQuaternion::from_quaternion(na::Quaternion::new(
            half_theta.cos(),
            quat_axis.x,
            quat_axis.y,
            quat_axis.z,
        ))
    } else {
        na::UnitQuaternion::identity()
    };

    let mut v = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        let ssym_omega = skew_sym(omega.clone_owned());
        v += ssym_omega * (1.0 - theta.cos()) / (theta.powi(2))
            + (ssym_omega * ssym_omega) * ((theta - theta.sin()) / (theta.powi(3)));
    }

    let trans = na::Translation::from(v * t);

    na::Isometry3::from_parts(trans, quat)
}

/// Converts an NAlgebra Isometry to a 6-Vector Lie Algebra representation
/// of a rigid body transform.
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn log_map(input: &na::Isometry3<f64>) -> na::Vector6<f64> {
    let t: na::Vector3<f64> = input.translation.vector;

    let quat = input.rotation;
    let theta: f64 = 2.0 * (quat.scalar()).acos();
    let half_theta = 0.5 * theta;
    let mut omega = na::Vector3::<f64>::zeros();

    let mut v_inv = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        omega = quat.vector() * theta / (half_theta.sin());
        let ssym_omega = skew_sym(omega);
        v_inv -= ssym_omega * 0.5;
        v_inv += ssym_omega * ssym_omega * (1.0 - half_theta * half_theta.cos() / half_theta.sin())
            / (theta * theta);
    }

    let mut ret = na::Vector6::<f64>::zeros();
    ret.fixed_view_mut::<3, 1>(0, 0).copy_from(&(v_inv * t));
    ret.fixed_view_mut::<3, 1>(3, 0).copy_from(&omega);

    ret
}

/// Produces the Jacobian of the exponential map of a lie algebra transform
/// that is then applied to a point with respect to the transform.
///
/// i.e.
/// d exp(t) * p
/// ------------
///      d t
///
/// The parameter 'transformed_point' is assumed to be transformed already
/// and thus: transformed_point = exp(t) * p
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn exp_map_jacobian(transformed_point: &na::Point3<f64>) -> na::Matrix3x6<f64> {
    let mut ss = na::Matrix3x6::zeros();
    ss.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&na::Matrix3::<f64>::identity());
    ss.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew_sym(transformed_point.coords)));
    ss
}

/// Projects a point in camera coordinates into the image plane
/// producing a floating-point pixel value
fn project(
    params: &Vector8<f64>, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    pt: &na::Point3<f64>,
) -> na::Point2<f64> {
    let fx = params[0];
    let fy = params[1];
    let cx = params[2];
    let cy = params[3];
    let k1 = params[4];
    let k2 = params[5];
    let p1 = params[6];
    let p2 = params[7];

    let xn = pt.x / pt.z;
    let yn = pt.y / pt.z;
    let rn2 = xn * xn + yn * yn;
    na::Point2::<f64>::new(
        fx * (xn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p1 * xn * yn + p2 * (rn2 + 2.0 * xn * xn)) + cx,
        fy * (yn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p2 * xn * yn + p1 * (rn2 + 2.0 * yn * yn)) + cy
    )
}

/// Jacobian of the projection function with respect to the four camera
/// paramaters (fx, fy, cx, cy). The 'transformed_pt' is a point already in
/// or transformed to camera coordinates.
fn proj_jacobian_wrt_params(transformed_pt: &na::Point3<f64>,
    params: &Vector8<f64> /*fx, fy, cx, cy, k1, k2, p1, p2*/) -> Matrix2x8<f64> {
    let fx = params[0];
    let fy = params[1];
    let k1 = params[4];
    let k2 = params[5];
    let p1 = params[6];
    let p2 = params[7];

    let xn = transformed_pt.x / transformed_pt.z;
    let yn = transformed_pt.y / transformed_pt.z;
    let rn2 = xn * xn + yn * yn;

    Matrix2x8::<f64>::from_row_slice(&[
        2.0 * p1 * xn * yn + p2 * (rn2 + 2.0 * xn * xn) + xn * (k1 * rn2 + k2 * rn2 * rn2 + 1.0),
        0.0,
        1.0,
        0.0,
        fx * rn2 * xn,
        fx * rn2 * rn2 * xn,
        2.0 * fx * xn * yn,
        fx * (rn2 + 2.0 * xn * xn),

        0.0,
        p1 * (rn2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn + yn * (k1 * rn2 + k2 * rn2 * rn2 + 1.0),
        0.0,
        1.0,
        fy * rn2 * yn,
        fy * rn2 * rn2 * yn,
        fy * (rn2 + 2.0 * yn * yn),
        2.0 * fy * xn * yn
    ])
}

/// Jacobian of the projection function with respect to the 3D point in camera
/// coordinates. The 'transformed_pt' is a point already in
/// or transformed to camera coordinates.
fn proj_jacobian_wrt_point(
    camera_model: &Vector8<f64>, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    transformed_pt: &na::Point3<f64>,
) -> na::Matrix2x3<f64> {
    let fx = camera_model[0];
    let fy = camera_model[1];
    let k1 = camera_model[4];
    let k2 = camera_model[5];
    let p1 = camera_model[6];
    let p2 = camera_model[7];

    let xn = transformed_pt.x / transformed_pt.z;
    let yn = transformed_pt.y / transformed_pt.z;
    let rn2 = xn * xn + yn * yn;
    let jacobian1 = na::Matrix2::<f64>::new(
        fx * (k1 * rn2 + k2 * rn2 * rn2 + 2.0 * p1 * yn + 4.0 * p2 * xn + 1.0),
        2.0 * fx * p1 * xn,
        2.0 * fy * p2 * yn,
        fy * (k1 * rn2 + k2 * rn2 * rn2 + 4.0 * p1 * yn + 2.0 * p2 * xn + 1.0),
    );
    let jacobian2 = na::Matrix2x3::<f64>::new(
        1.0 / transformed_pt.z,
        0.0,
        -transformed_pt.x / (transformed_pt.z.powi(2)),
        0.0,
        1.0 / transformed_pt.z,
        -transformed_pt.y / (transformed_pt.z.powi(2)),
    );
    jacobian1 * jacobian2
}

/// Struct for holding data for calibration.
struct Calibration<'a> {
    model_pts: &'a Vec<na::Point3<f64>>,
    image_pts_set: &'a Vec<Vec<na::Point2<f64>>>,
}

impl<'a> Calibration<'a> {
    /// Decode the camera model and transforms from the flattened parameter vector
    ///
    /// The convention in use is that the first four values are the camera parameters
    /// and each set of six values after is a transform (one per image). This convention
    /// is also followed in the Jacobian function below.
    fn decode_params(
        &self,
        param: &na::DVector<f64>,
    ) -> (Vector8<f64>, Vec<na::Isometry3<f64>>) {
        // Camera parameters are in the first four elements
        let camera_model: Vector8<f64> = param.fixed_view::<8, 1>(0, 0).clone_owned();

        // Following the camera parameters, for each image there
        // will be one 6D transform
        let transforms = self
            .image_pts_set
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let lie_alg_transform: na::Vector6<f64> =
                    param.fixed_view::<6, 1>(8 + 6 * i, 0).clone_owned();
                // Convert to a useable na::Isometry
                exp_map(&lie_alg_transform)
            })
            .collect::<Vec<_>>();
        (camera_model, transforms)
    }
}


impl lm::LMProblem for Calibration<'_> {
    fn residual(&self, param: &na::DVector<f64>) -> na::DVector<f64> {
                // Get usable camera model and transforms from the parameter vector
        let (camera_model, transforms) = self.decode_params(param);

        let num_images = self.image_pts_set.len();
        let num_target_points = self.model_pts.len();
        let num_residuals = num_images * num_target_points;

        // Allocate big empty residual
        let mut residual = na::DVector::<f64>::zeros(num_residuals * 2);

        let mut residual_idx = 0;
        for (image_pts, transform) in self.image_pts_set.iter().zip(transforms.iter()) {
            for (observed_image_pt, target_pt) in image_pts.iter().zip(self.model_pts.iter()) {
                // Apply image formation model
                let transformed_point = transform * target_pt;
                let projected_pt = project(&camera_model, &transformed_point);

                // Populate residual vector two rows at time
                let individual_residual = projected_pt - observed_image_pt;
                residual
                    .fixed_view_mut::<2, 1>(residual_idx, 0)
                    .copy_from(&individual_residual);
                residual_idx += 2;
            }
        }

        residual
    }


    fn jacobian(&self, param: &na::DVector<f64>) -> na::DMatrix<f64> {
        // Get usable camera model and transforms from the parameter vector
        let (camera_model, transforms) = self.decode_params(param);

        let num_images = self.image_pts_set.len();
        let num_target_points = self.model_pts.len();
        let num_residuals = num_images * num_target_points;
        let num_unknowns = 6 * num_images + 8;

        // Allocate big empty Jacobian
        let mut jacobian = na::DMatrix::<f64>::zeros(num_residuals * 2, num_unknowns);

        let mut residual_idx = 0;
        for (tform_idx, transform) in transforms.iter().enumerate() {
            for target_pt in self.model_pts.iter() {
                // Apply image formation model
                let transformed_point = transform * target_pt;

                // Populate Jacobian matrix two rows at time

                // Populate Jacobian part for the camera parameters
                jacobian
                    .fixed_view_mut::<2, 8>(
                        residual_idx,
                        0, /*first four columns are camera parameters*/
                    )
                    .copy_from(&proj_jacobian_wrt_params(&transformed_point, &camera_model));

                // Populate the Jacobian part for the transform
                let proj_jacobian_wrt_point =
                    proj_jacobian_wrt_point(&camera_model, &transformed_point);
                let transform_jacobian_wrt_transform = exp_map_jacobian(&transformed_point);

                // Transforms come after camera parameters in sets of six columns
                jacobian
                    .fixed_view_mut::<2, 6>(residual_idx, 8 + tform_idx * 6)
                    .copy_from(&(proj_jacobian_wrt_point * transform_jacobian_wrt_transform));

                residual_idx += 2;
            }
        }
        jacobian
    }
}

use std::fs::OpenOptions;
use std::io::prelude::*;

pub fn optimize_with_lm() -> Result<(), Box<dyn std::error::Error>> {
    let (k, 
        mut tfs, 
        img_points_set, 
        world_points) = crate::calibrate::calibrate((11, 8))?;
    
    // save optimization problem
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open("./problem.json")
        .unwrap();
    let k_json = serde_json::to_string(&k)?;
    let tfs_json = serde_json::to_string(&tfs)?;
    let img_points_set_json = serde_json::to_string(&img_points_set)?;
    let world_points_json = serde_json::to_string(&world_points)?;

    writeln!(file, "{{\"K\":\n{}\n,", k_json)?;
    writeln!(file, "\"world_points\":\n{}\n,", world_points_json)?;
    writeln!(file, "\"extrinsics\":\n{}\n,", tfs_json)?;
    writeln!(file, "\"image_points\":\n{}\n}},", img_points_set_json)?;

    let world_points: Vec<_> = world_points.iter().map(|p| {
        na::Point3::<f64>::new(p.x, p.y, 0.0)
    }).collect();

    // Create calibration parameters
    let cal_cost = Calibration {
        model_pts: &world_points,
        image_pts_set: &img_points_set,
    };
    let mut init_param = na::DVector::<f64>::zeros(8 + img_points_set.len() * 6);
    init_param[0] = k[(0, 0)];
    init_param[1] = k[(1, 1)];
    init_param[2] = k[(0, 2)];
    init_param[3] = k[(1, 2)];
    init_param[4] = 0.0;
    init_param[5] = 0.0;
    init_param[6] = 0.0;
    init_param[7] = 0.0;

    for (idx, tf) in tfs.iter().enumerate() {
        init_param
            .fixed_view_mut::<6, 1>(8 + 6 * idx, 0)
            .copy_from(&log_map(tf));
    }

    let max_iter = 100;
    let tol = 1e-8;
    let lambda = 1e-3;
    let gamma = 10.0;

    let res = crate::lm::levenberg_marquardt(cal_cost, init_param, max_iter, tol, lambda, gamma);

    eprintln!("{}\n\n", res);

    // Print intrinsics results
    eprintln!("ground truth intrinsics: {:?}", vec![k[(0,0)], k[(1,1)], k[(0,2)], k[(1,2)]]);

    eprintln!(
        "optimized intrinsics: {}",
        res.fixed_view::<8, 1>(0, 0)
    );

    // Print transforms
    for (i, t) in tfs.iter_mut().enumerate() {
            *t = exp_map(
                    &res
                    .fixed_view::<6, 1>(8 + 6 * i, 0)
                    .clone_owned()
            );
    }


    // show images
    let project = |K: na::Matrix3<f64>, tf: na::Isometry3<f64>, world_point: na::Point3<f64>|
    -> na::Point2<f64> {
        let transed = tf * world_point;
        let xn = transed.x / transed.z;
        let yn = transed.y / transed.z;
        let fx = K[(0, 0)];
        let fy = K[(1, 1)];
        let cx = K[(0, 2)];
        let cy = K[(1, 2)];
        let x = fx * xn + cx;
        let y = fy * yn + cy;
        na::Point2::<f64>::new(x, y)
    };
    
    let paths: Vec<String> = (0..=40).into_iter().map(|x| format!("./cali/{}.png", 100000 + x)).collect();
    
    for (idx, path) in paths.iter().enumerate() {
        let mut image = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
        for (idx_p, img_point) in img_points_set[idx].iter().enumerate() {
            
            let wp = na::Point3::<f64>::new(world_points[idx_p].x, world_points[idx_p].y, 0.0);
            let repro_p = project(k, tfs[idx], wp); 
            imgproc::circle(&mut image,
                core::Point2i::new(img_point.x as i32, img_point.y as i32),
                5,
                core::Scalar::new(0.0, 255.0, 0.0, 255.0),
                1,
                imgproc::LINE_8, 0)?;

            imgproc::circle(&mut image,
                core::Point2i::new(repro_p.x as i32, repro_p.y as i32),
                5,
                core::Scalar::new(255.0, 0.0, 0.0, 255.0),
                1,
                imgproc::LINE_8, 0)?;
            let x_p = project(k, tfs[idx], na::Point3::<f64>::new(0.2, 0.0, 0.0));
            let y_p = project(k, tfs[idx], na::Point3::<f64>::new(0.0, 0.14, 0.0));
            let z_p = project(k, tfs[idx], na::Point3::<f64>::new(0.0, 0.0, 0.2));
            let o_p = project(k, tfs[idx], na::Point3::<f64>::new(0.0, 0.0, 0.0));
            imgproc::line(&mut image,
                core::Point2i::new(o_p.x as i32, o_p.y as i32),
                core::Point2i::new(x_p.x as i32, x_p.y as i32),
                core::Scalar::new(255.0, 0.0, 0.0, 255.0),
                3,
                imgproc::LINE_8,
                0)?;
            imgproc::line(&mut image,
                core::Point2i::new(o_p.x as i32, o_p.y as i32),
                core::Point2i::new(y_p.x as i32, y_p.y as i32),
                core::Scalar::new(0.0, 255.0, 0.0, 255.0),
                3,
                imgproc::LINE_8,
                0)?;
            imgproc::line(&mut image,
                core::Point2i::new(o_p.x as i32, o_p.y as i32),
                core::Point2i::new(z_p.x as i32, z_p.y as i32),
                core::Scalar::new(0.0, 0.0, 255.0, 255.0),
                3,
                imgproc::LINE_8,
                0)?;
        }
        // println!("{}:\n", idx);
        // println!("{}", tfs[idx].to_matrix());
        crate::gui::imshow("title", &image)?;
    }



    Ok(())
}


#[cfg(test)]
mod test {
    use std::error::Error;

    #[test]
    fn test_optimize_with_lm() -> Result<(), Box<dyn Error>> {
        super::optimize_with_lm()?;

        Err(From::from("end"))
    }
    

}
