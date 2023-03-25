#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use egui_extras::RetainedImage;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(600.0, 400.0)),
        ..Default::default()
    };

    eframe::run_native(
        "image",
        options,
        Box::new(|_cc| Box::<MyApp>::new(MyApp::new("/home/zhang/Pictures/cali.png"))),
    )
}

struct MyApp {
    image: RetainedImage,
    ratio: f32,
    zoom: f32,
    offset: egui::Vec2,
}

impl MyApp {
    fn new(path: &str) -> Self {
        use std::fs::read;
        let image = RetainedImage::from_image_bytes(
                "cali.png",
                &read(path).unwrap(),
            )
            .unwrap();
        let ratio = image.width() as f32 / image.height() as f32;
        Self {
            image,
            ratio,
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
        }
    }
}

impl Default for MyApp {
    fn default() -> Self {
        let image = RetainedImage::from_image_bytes(
                "cali.png",
                include_bytes!("/home/zhang/Pictures/cali.png"),
            )
            .unwrap();
        let ratio = image.width() as f32 / image.height() as f32;
        Self {
            image,
            ratio,
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size();

            let mut image_size = if available_size.x / available_size.y > self.ratio {
                            egui::vec2(available_size.y * self.ratio, available_size.y)
                        } else {
                            egui::vec2(available_size.x, available_size.x / self.ratio)
                        };

            let scroll_delta = ctx.input(|i| i.scroll_delta).y;
            if scroll_delta != 0.0 {
                // 根据滚轮增量更新缩放因子
                let zoom_speed = 0.001; // 设置缩放速度
                self.zoom += scroll_delta * zoom_speed;
                self.zoom = self.zoom.max(0.1); // 限制最小缩放比例
            }
            image_size *= self.zoom;
            

            let image_rect = egui::Rect::from_min_size(ui.min_rect().center() - image_size * 0.5 + self.offset, image_size);
            let image = egui::Image::new(self.image.texture_id(ctx), image_size);
                
            ui.allocate_space(image_rect.size()); // 为图片分配空间
            image.paint_at(ui, image_rect);

            // 处理拖动事件
            let id = ui.make_persistent_id("image_drag");
            let response = ui.interact(image_rect, id, egui::Sense::drag());
            if response.dragged() {
                self.offset += 0.75 * ctx.input(|i| i.pointer.delta());
            }
        });
    }
}