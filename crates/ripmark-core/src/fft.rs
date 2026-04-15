//! 2-D FFT helpers wrapping `rustfft`.
//!
//! Images are f32, row-major, channel-last (H x W x C).
//! Spectra are row-major complex arrays (H x W).

use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

/// A 2-D complex spectrum in standard FFT layout (zero-frequency at [0,0]).
pub struct Spectrum {
    pub data: Array2<Complex<f32>>,
}

fn planner() -> &'static Mutex<FftPlanner<f32>> {
    static PLANNER: OnceLock<Mutex<FftPlanner<f32>>> = OnceLock::new();
    PLANNER.get_or_init(|| Mutex::new(FftPlanner::<f32>::new()))
}

fn fft_plan(len: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    let mut planner = planner().lock().expect("fft planner mutex poisoned");
    if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    }
}

impl Spectrum {
    pub fn height(&self) -> usize {
        self.data.nrows()
    }
    pub fn width(&self) -> usize {
        self.data.ncols()
    }

    /// Element-wise magnitude.
    pub fn magnitude(&self) -> Array2<f32> {
        self.data.mapv(|c| c.norm())
    }

    /// Element-wise phase in [-pi, pi].
    pub fn phase(&self) -> Array2<f32> {
        self.data.mapv(|c| c.arg())
    }

    /// Shift zero-frequency to the centre (equivalent to numpy.fftshift).
    pub fn shifted(self) -> Self {
        Self {
            data: fftshift(self.data),
        }
    }

    /// Undo a previous shift (equivalent to numpy.ifftshift).
    pub fn unshifted(self) -> Self {
        Self {
            data: ifftshift(self.data),
        }
    }

    /// Inverse FFT, returning the real part of the spatial output.
    pub fn to_spatial(self) -> Array2<f32> {
        ifft2(self.data)
    }
}

/// 2-D FFT of a single real-valued channel. Equivalent to scipy.fft.fft2.
pub fn fft2(input: ArrayView2<f32>) -> Spectrum {
    let h = input.nrows();
    let w = input.ncols();

    let mut buf: Vec<Complex<f32>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Forward FFT along rows.
    let row_plan = fft_plan(w, false);
    for row in buf.chunks_mut(w) {
        row_plan.process(row);
    }

    // Forward FFT along columns: extract each column, transform, put back.
    let col_plan = fft_plan(h, false);
    let mut col = vec![Complex::new(0.0f32, 0.0); h];
    for c in 0..w {
        for r in 0..h {
            col[r] = buf[r * w + c];
        }
        col_plan.process(&mut col);
        for r in 0..h {
            buf[r * w + c] = col[r];
        }
    }

    Spectrum {
        data: Array2::from_shape_vec((h, w), buf).expect("shape matches buffer length"),
    }
}

/// 2-D FFT of each RGB channel independently.
/// Returns [R_spectrum, G_spectrum, B_spectrum].
pub fn fft2_rgb(image: ArrayView3<f32>) -> [Spectrum; 3] {
    std::array::from_fn(|c| fft2(image.slice(ndarray::s![.., .., c])))
}

/// Convert an RGB image to grayscale, then compute its 2-D FFT.
pub fn fft2_gray(image: ArrayView3<f32>) -> Spectrum {
    fft2(to_grayscale(image).view())
}

/// Shift zero-frequency to the centre of the spectrum (numpy.fftshift).
/// Element at [i, j] moves to [(i + H/2) % H, (j + W/2) % W].
pub fn fftshift(arr: Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let (h, w) = arr.dim();
    let sh = h / 2;
    let sw = w / 2;
    Array2::from_shape_fn((h, w), |(i, j)| arr[[(i + sh) % h, (j + sw) % w]])
}

/// Inverse of fftshift (numpy.ifftshift).
/// For even dimensions this is identical to fftshift.
/// For odd dimensions it shifts by ceil(n/2) rather than floor(n/2).
pub fn ifftshift(arr: Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let (h, w) = arr.dim();
    let sh = (h + 1) / 2;
    let sw = (w + 1) / 2;
    Array2::from_shape_fn((h, w), |(i, j)| arr[[(i + sh) % h, (j + sw) % w]])
}

/// fftshift for real-valued arrays (magnitude or phase maps).
pub fn fftshift_real(arr: Array2<f32>) -> Array2<f32> {
    let (h, w) = arr.dim();
    let sh = h / 2;
    let sw = w / 2;
    Array2::from_shape_fn((h, w), |(i, j)| arr[[(i + sh) % h, (j + sw) % w]])
}

/// Convert RGB (H x W x 3, values in [0, 1]) to grayscale.
/// Uses ITU-R BT.601 luminance: Y = 0.299*R + 0.587*G + 0.114*B.
pub fn to_grayscale(image: ArrayView3<f32>) -> Array2<f32> {
    let h = image.shape()[0];
    let w = image.shape()[1];
    Array2::from_shape_fn((h, w), |(y, x)| {
        0.299 * image[[y, x, 0]] + 0.587 * image[[y, x, 1]] + 0.114 * image[[y, x, 2]]
    })
}

/// Bilinear resize of a grayscale image to (out_h, out_w).
pub fn resize_gray(src: &Array2<f32>, out_h: usize, out_w: usize) -> Array2<f32> {
    let (src_h, src_w) = src.dim();
    let scale_y = src_h as f32 / out_h as f32;
    let scale_x = src_w as f32 / out_w as f32;

    Array2::from_shape_fn((out_h, out_w), |(oy, ox)| {
        let sy = (oy as f32 + 0.5) * scale_y - 0.5;
        let sx = (ox as f32 + 0.5) * scale_x - 0.5;

        let y0 = (sy.floor() as isize).clamp(0, src_h as isize - 1) as usize;
        let x0 = (sx.floor() as isize).clamp(0, src_w as isize - 1) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let x1 = (x0 + 1).min(src_w - 1);

        let fy = sy - sy.floor();
        let fx = sx - sx.floor();

        src[[y0, x0]] * (1.0 - fy) * (1.0 - fx)
            + src[[y0, x1]] * (1.0 - fy) * fx
            + src[[y1, x0]] * fy * (1.0 - fx)
            + src[[y1, x1]] * fy * fx
    })
}

/// Bilinear resize of an RGB image (H x W x 3) to (out_h, out_w).
pub fn resize_rgb(src: &Array3<f32>, out_h: usize, out_w: usize) -> Array3<f32> {
    let (src_h, src_w, _) = src.dim();
    let scale_y = src_h as f32 / out_h as f32;
    let scale_x = src_w as f32 / out_w as f32;

    Array3::from_shape_fn((out_h, out_w, 3), |(oy, ox, c)| {
        let sy = (oy as f32 + 0.5) * scale_y - 0.5;
        let sx = (ox as f32 + 0.5) * scale_x - 0.5;

        let y0 = (sy.floor() as isize).clamp(0, src_h as isize - 1) as usize;
        let x0 = (sx.floor() as isize).clamp(0, src_w as isize - 1) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let x1 = (x0 + 1).min(src_w - 1);

        let fy = sy - sy.floor();
        let fx = sx - sx.floor();

        src[[y0, x0, c]] * (1.0 - fy) * (1.0 - fx)
            + src[[y0, x1, c]] * (1.0 - fy) * fx
            + src[[y1, x0, c]] * fy * (1.0 - fx)
            + src[[y1, x1, c]] * fy * fx
    })
}

// 2-D inverse FFT, normalised by 1/(H*W). Returns real part.
fn ifft2(data: Array2<Complex<f32>>) -> Array2<f32> {
    let h = data.nrows();
    let w = data.ncols();
    let mut buf: Vec<Complex<f32>> = data.into_iter().collect();

    let row_plan = fft_plan(w, true);
    for row in buf.chunks_mut(w) {
        row_plan.process(row);
    }

    let col_plan = fft_plan(h, true);
    let mut col = vec![Complex::new(0.0f32, 0.0); h];
    for c in 0..w {
        for r in 0..h {
            col[r] = buf[r * w + c];
        }
        col_plan.process(&mut col);
        for r in 0..h {
            buf[r * w + c] = col[r];
        }
    }

    let norm = (h * w) as f32;
    Array2::from_shape_fn((h, w), |(r, c)| buf[r * w + c].re / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn roundtrip_fft_ifft() {
        let input = Array2::from_shape_fn((16, 16), |(r, c)| {
            (2.0 * PI * r as f32 / 16.0).sin() + (2.0 * PI * c as f32 / 8.0).cos()
        });
        let recovered = fft2(input.view()).to_spatial();
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "round-trip error: {a} vs {b}");
        }
    }

    #[test]
    fn fftshift_ifftshift_identity() {
        let original = Array2::from_shape_fn((8, 8), |(r, c)| Complex::new(r as f32, c as f32));
        let result = ifftshift(fftshift(original.clone()));
        for (a, b) in original.iter().zip(result.iter()) {
            assert!((a.re - b.re).abs() < 1e-6);
            assert!((a.im - b.im).abs() < 1e-6);
        }
    }

    #[test]
    fn dc_at_origin_then_centre_after_shift() {
        let constant = Array2::from_elem((8, 8), 1.0_f32);
        let spec = fft2(constant.view());
        assert!(spec.data[[0, 0]].re > 1.0);
        let shifted = spec.shifted();
        assert!(shifted.data[[4, 4]].re > 1.0);
    }

    #[test]
    fn grayscale_of_grey_image_is_identity() {
        let v = 0.6_f32;
        let img = Array3::from_elem((4, 4, 3), v);
        let gray = to_grayscale(img.view());
        for &p in gray.iter() {
            assert!((p - v).abs() < 1e-5);
        }
    }
}
