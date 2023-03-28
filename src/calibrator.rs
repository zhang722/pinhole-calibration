use std::error::Error;

use image::{GrayImage};

#[derive(Debug)]
pub struct Harris {
    pub x: i32,
    pub y: i32,
    pub score: f32,
}

pub fn calibrator() {

}

pub fn nms(scores: &mut [f32], img_width: usize, img_height: usize, window_size: usize, nms_radius: usize) {
    let half_win_size = window_size / 2;
    for y in half_win_size..img_height - half_win_size {
        for x in half_win_size..img_width - half_win_size {
            let index = y * img_width + x;
            let score = scores[index];
            let mut is_maximum = true;
            for wy in -(nms_radius as i32)..(nms_radius as i32) + 1 {
                for wx in -(nms_radius as i32)..(nms_radius as i32) + 1 {
                    if wx == 0 && wy == 0 {
                        continue;
                    }
                    let px = x as i32 + wx;
                    let py = y as i32 + wy;
                    if px < half_win_size as i32
                        || px >= img_width as i32 - half_win_size as i32
                        || py < half_win_size as i32
                        || py >= img_height as i32 - half_win_size as i32
                    {
                        continue;
                    }
                    let idx = py as usize * img_width + px as usize;
                    if score <= scores[idx] {
                        is_maximum = false;
                        scores[index] = 0.;
                        break;
                    }
                }
                if !is_maximum {
                    break;
                }
            }
        }
    }
}

pub fn detect_harris(img: &GrayImage, threshold: u8) -> Result<Vec<Harris>, Box<dyn Error>> {
    todo!()
}

#[cfg(test)]
mod test{
    use imageproc::{drawing};
    use std::error::Error;

    #[test]
    fn test_detect_corners() -> Result<(), Box<dyn Error>> {
        let mut gray = image::open("/home/zhang/Pictures/cali.png")?.into_luma8();

        let hollow_circle_color = image::Luma([100]); // Green
        let corners = imageproc::corners::corners_fast9(&gray, 100);
        println!("cornre num:{}", corners.len());
        assert!(corners.len() > 10);
        for corner in corners {
            drawing::draw_hollow_circle_mut(&mut gray, (corner.x as i32, corner.y as i32), 10, hollow_circle_color);
        }

        let output_path = "/home/zhang/Pictures/cali_with_corners.png";
        gray.save(output_path).unwrap();

        Ok(())
    }

    fn nms(scores: &[f32]) -> Vec<(i32, i32)> {
        todo!()
    }

    #[test]
    fn test_harris() -> Result<(), Box<dyn Error>> {
        const IN_PATH: &str = "cali.png";
        const OUT_PATH: &str = "cali_harris.png";
        const WIN_SIZE: usize = 3;
        const HALF_WIN_SIZE: usize = WIN_SIZE / 2;

        let input = image::open(IN_PATH)?;
        let mut rgb = input.to_rgb8();
        let img = input.to_luma8();
        let img_width = img.width() as usize;
        let img_height = img.height() as usize;

        let sobel_x = imageproc::gradients::horizontal_sobel(&img);
        let sobel_y = imageproc::gradients::vertical_sobel(&img);

        let mut pixel_avg = vec![0.0; img_width * img_height];

        for y in HALF_WIN_SIZE..img_height - WIN_SIZE {
            for x in HALF_WIN_SIZE..img_width - WIN_SIZE {
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;

                // compute average of gradient
                for wy in -(HALF_WIN_SIZE as isize)..HALF_WIN_SIZE as isize + 1 {
                    for wx in -(HALF_WIN_SIZE as isize)..HALF_WIN_SIZE as isize + 1 {
                        let px = (x as isize + wx) as u32;
                        let py = (y as isize + wy) as u32;
                        let gx = sobel_x.get_pixel(px, py)[0] as f32;
                        let gy = sobel_y.get_pixel(px, py)[0] as f32;
                        sum_x += gx;
                        sum_y += gy;
                    }
                }
                let idx = y * img_width + x;
                let pixel_sum = (sum_x * sum_x + sum_y * sum_y).sqrt();
                pixel_avg[idx] = pixel_sum / (WIN_SIZE * WIN_SIZE) as f32;
            }
        }

        // walk through again, get the score
        let mut scores = vec![0.0; img_width * img_height];
        for y in HALF_WIN_SIZE..img_height - WIN_SIZE {
            for x in HALF_WIN_SIZE..img_width - WIN_SIZE {
                let mut m11 = 0.0;
                let mut m22 = 0.0;
                let mut m12 = 0.0;

                // compute average of gradient
                for wy in -(HALF_WIN_SIZE as isize)..HALF_WIN_SIZE as isize + 1 {
                    for wx in -(HALF_WIN_SIZE as isize)..HALF_WIN_SIZE as isize + 1 {
                        let px = (x as isize + wx) as u32;
                        let py = (y as isize + wy) as u32;
                        let gx = sobel_x.get_pixel(px, py)[0] as f32;
                        let gy = sobel_y.get_pixel(px, py)[0] as f32;
                        let idx = (py * img_width as u32 + px) as usize;
                        let avg = pixel_avg[idx];
                        m11 += (gx - avg) * (gx - avg);
                        m22 += (gy - avg) * (gy - avg);
                        m12 += (gx - avg) * (gy - avg);
                    }
                }
                let det = m11 * m22 - m12 * m12;
                let trace = m11 + m12;
                let k = 0.05;

                let idx = y * img_width + x;
                scores[idx] = det - k * trace * trace;
            }
        }

        let mut corners: Vec<crate::calibrator::Harris> = Vec::new();
        let threshold = 10000000.0;
        let nms_radius = 10;

        for score in scores.iter_mut() {
            if *score < threshold {
                *score = 0.;
            }
        }
    
        crate::calibrator::nms(&mut scores, img_width, img_height, WIN_SIZE, nms_radius);

        for x in HALF_WIN_SIZE.. img_width - HALF_WIN_SIZE {
            for y in HALF_WIN_SIZE.. img_height - HALF_WIN_SIZE {
                let idx = y * img_width + x;
                let score = scores[idx];
                if scores > 0. {
                    corners.push(crate::calibrator::Harris {x: x as i32, y: y as i32, score});
                }
            }
        }

        for corner in corners {
            println!("{:?}", corner);
            imageproc::drawing::draw_hollow_circle_mut(&mut rgb, (corner.x, corner.y), 5, image::Rgb([255u8, 0u8, 0]));
        }
        rgb.save(OUT_PATH)?;

        Err(From::from("end"))
    }

    #[test]
    fn test_draw_circle() -> Result<(), Box<dyn Error>> {
        use image::{Rgb};

        let mut img = image::open("cali.png")?.into_rgb8();
        let center = (256, 256);
        let radius = 50;
        let color = Rgb([255u8, 0, 0]); // red
        imageproc::drawing::draw_hollow_circle_mut(&mut img, center, radius, color);
        img.save("test.png")?;

        Err(From::from("end"))
    }
    
}