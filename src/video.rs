//! 每个相机一个后台线程：把帧送入 ffmpeg 编码为 LeRobot 视频文件，并对抽样帧计算图像统计

use anyhow::{Context, Result, bail, ensure};
use indicatif::ProgressBar;
use serde_json::{Map, Value as Json, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::JoinHandle,
};

use crate::config::VideoConfig;
use crate::stats::FeatureStats;

/// 原始图像（sensor_msgs/Image）支持的像素格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawFormat {
    Rgb8,
    Bgr8,
    Mono8,
    /// 平面 YUV 4:2:0，来自 H.264 解码
    Yuv420p,
}

impl RawFormat {
    pub fn from_encoding(encoding: &str) -> Result<Self> {
        Ok(match encoding {
            "rgb8" => Self::Rgb8,
            "bgr8" => Self::Bgr8,
            "mono8" | "8UC1" => Self::Mono8,
            other => bail!("暂不支持的图像编码 {other}（支持 rgb8、bgr8、mono8）"),
        })
    }

    pub fn channels(self) -> usize {
        match self {
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Mono8 => 1,
            Self::Yuv420p => unreachable!("yuv420p 不是 sensor_msgs/Image 的编码"),
        }
    }

    fn ffmpeg_pix_fmt(self) -> &'static str {
        match self {
            Self::Rgb8 => "rgb24",
            Self::Bgr8 => "bgr24",
            Self::Mono8 => "gray",
            Self::Yuv420p => "yuv420p",
        }
    }
}

/// 送入编码器的一帧
#[derive(Clone)]
pub enum Frame {
    /// JPEG / PNG 压缩图像
    Encoded(Arc<Vec<u8>>),
    /// 紧密排列（无行填充）的原始像素
    Raw { data: Arc<Vec<u8>>, width: u32, height: u32, format: RawFormat },
}

impl Frame {
    fn describe(&self) -> String {
        match self {
            Frame::Encoded(data) => match image::guess_format(data) {
                Ok(f) => format!("{f:?}"),
                Err(_) => "未知压缩格式".into(),
            },
            Frame::Raw { width, height, format, .. } => format!("{format:?} {width}x{height}"),
        }
    }

    /// 解码为 RGB（HWC）
    fn decode_rgb(&self) -> Result<(u32, u32, Vec<u8>)> {
        match self {
            Frame::Encoded(data) => {
                let img = image::load_from_memory(data).context("解码图像失败")?.into_rgb8();
                Ok((img.width(), img.height(), img.into_raw()))
            }
            Frame::Raw { data, width, height, format } => {
                let rgb = match format {
                    RawFormat::Rgb8 => data.to_vec(),
                    RawFormat::Bgr8 => data.chunks_exact(3).flat_map(|p| [p[2], p[1], p[0]]).collect(),
                    RawFormat::Mono8 => data.iter().flat_map(|&v| [v, v, v]).collect(),
                    RawFormat::Yuv420p => yuv420p_to_rgb(data, *width as usize, *height as usize),
                };
                Ok((*width, *height, rgb))
            }
        }
    }
}

enum Msg {
    Episode { index: usize, length: usize },
    Frame(Frame),
}

/// 一个 episode 在视频文件中的位置
#[derive(Debug, Clone)]
pub struct EpisodeVideo {
    pub episode_index: usize,
    pub chunk_index: usize,
    pub file_index: usize,
    pub from_timestamp: f64,
    pub to_timestamp: f64,
}

pub struct CameraResult {
    pub episodes: Vec<EpisodeVideo>,
    /// 与 episodes 一一对应
    pub stats: Vec<FeatureStats>,
    pub width: u32,
    pub height: u32,
    /// info.json 中该 feature 的 info 字段
    pub info: Map<String, Json>,
}

/// 视频文件布局参数
#[derive(Clone)]
pub struct VideoLayout {
    pub root: PathBuf,
    pub video_key: String,
    pub fps: u32,
    pub chunks_size: usize,
    pub file_size_mb: f64,
}

impl VideoLayout {
    pub fn path(&self, chunk_index: usize, file_index: usize) -> PathBuf {
        self.root
            .join("videos")
            .join(&self.video_key)
            .join(format!("chunk-{chunk_index:03}/file-{file_index:03}.mp4"))
    }
}

pub struct CameraEncoder {
    key: String,
    tx: Option<SyncSender<Msg>>,
    handle: Option<JoinHandle<Result<CameraResult>>>,
}

impl CameraEncoder {
    /// `bar` 在每帧写入 ffmpeg 后前进，长度由调用方设置
    pub fn spawn(layout: VideoLayout, encoder: VideoConfig, bar: ProgressBar) -> Self {
        // 解码后的原始帧每帧数 MB，缓冲不宜过大
        let (tx, rx) = sync_channel(32);
        let key = layout.video_key.clone();
        let handle = std::thread::Builder::new()
            .name(format!("video {key}"))
            .spawn(move || Worker::new(layout, encoder, bar).run(rx))
            .expect("创建视频编码线程失败");
        Self { key, tx: Some(tx), handle: Some(handle) }
    }

    /// 开始一个新的 episode，之后的 `length` 帧属于它
    pub fn start_episode(&mut self, index: usize, length: usize) -> Result<()> {
        self.send(Msg::Episode { index, length })
    }

    pub fn push(&mut self, frame: Frame) -> Result<()> {
        self.send(Msg::Frame(frame))
    }

    fn send(&mut self, msg: Msg) -> Result<()> {
        let tx = self.tx.as_ref().expect("编码器已结束");
        if tx.send(msg).is_err() {
            // 线程已提前退出，取回它的错误
            return Err(self.join().err().unwrap_or_else(|| anyhow::anyhow!("视频编码线程意外退出")));
        }
        Ok(())
    }

    fn join(&mut self) -> Result<CameraResult> {
        self.tx = None;
        let handle = self.handle.take().context("编码器已结束")?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("视频编码线程 panic"))?
            .with_context(|| format!("相机 {} 编码失败", self.key))
    }

    pub fn finish(mut self) -> Result<CameraResult> {
        self.join()
    }
}

struct Ffmpeg {
    child: Child,
    stdin: ChildStdin,
    path: PathBuf,
}

impl Ffmpeg {
    fn spawn(frame: &Frame, path: &Path, fps: u32, encoder: &VideoConfig) -> Result<Self> {
        fs::create_dir_all(path.parent().unwrap())?;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
        match frame {
            Frame::Encoded(data) => {
                let codec = match image::guess_format(data) {
                    Ok(image::ImageFormat::Jpeg) => "mjpeg",
                    Ok(image::ImageFormat::Png) => "png",
                    _ => bail!("不支持的压缩图像格式：{}", frame.describe()),
                };
                cmd.args(["-f", "image2pipe", "-c:v", codec]);
            }
            Frame::Raw { width, height, format, .. } => {
                cmd.args(["-f", "rawvideo", "-pix_fmt", format.ffmpeg_pix_fmt()]);
                cmd.args(["-video_size", &format!("{width}x{height}")]);
            }
        }
        cmd.args(["-framerate", &fps.to_string(), "-i", "pipe:0"]);
        // 与 lerobot（PyAV 从 RGB 转换）一致输出 limited range 的 yuv
        cmd.args(["-vf", &format!("scale=out_range=tv,format={}", encoder.pix_fmt)]);
        cmd.args(["-c:v", &encoder.vcodec]);
        for (k, v) in encoder.codec_options() {
            cmd.args([format!("-{k}"), v]);
        }
        cmd.args(["-movflags", "+faststart"]).arg(path);
        cmd.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());

        let mut child = cmd.spawn().context("启动 ffmpeg 失败，请确认已安装 ffmpeg")?;
        let stdin = child.stdin.take().unwrap();
        Ok(Self { child, stdin, path: path.to_path_buf() })
    }

    fn write(&mut self, frame: &Frame) -> Result<()> {
        let data = match frame {
            Frame::Encoded(d) | Frame::Raw { data: d, .. } => d,
        };
        if self.stdin.write_all(data).is_ok() {
            return Ok(());
        }
        // 写入失败说明 ffmpeg 已退出，返回它的错误输出
        let Self { child, stdin, .. } = self;
        let _ = stdin.flush();
        bail!("ffmpeg 异常退出：{}", read_stderr(child))
    }

    fn finish(self) -> Result<()> {
        let Self { mut child, stdin, path } = self;
        drop(stdin);
        let stderr = read_stderr(&mut child);
        let status = child.wait()?;
        ensure!(status.success(), "ffmpeg 编码 {} 失败：{stderr}", path.display());
        Ok(())
    }
}

fn read_stderr(child: &mut Child) -> String {
    let mut s = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut s);
    }
    s.trim().to_string()
}

struct EpisodeState {
    index: usize,
    length: usize,
    received: usize,
    from_frame: usize,
    /// 需要抽样统计的帧号（episode 内），升序
    samples: Vec<usize>,
    next_sample: usize,
    pixel_counts: [[u64; 256]; 3],
}

struct Worker {
    layout: VideoLayout,
    bar: ProgressBar,
    encoder: VideoConfig,
    ffmpeg: Option<Ffmpeg>,
    chunk_index: usize,
    file_index: usize,
    frames_in_file: usize,
    size: Option<(u32, u32)>,
    /// 当前 ffmpeg 进程的输入格式；格式变化时需要换新文件
    ffmpeg_kind: Option<String>,
    episode: Option<EpisodeState>,
    result_episodes: Vec<EpisodeVideo>,
    result_stats: Vec<FeatureStats>,
}

impl Worker {
    fn new(layout: VideoLayout, encoder: VideoConfig, bar: ProgressBar) -> Self {
        Self {
            layout,
            bar,
            encoder,
            ffmpeg: None,
            chunk_index: 0,
            file_index: 0,
            frames_in_file: 0,
            size: None,
            ffmpeg_kind: None,
            episode: None,
            result_episodes: Vec::new(),
            result_stats: Vec::new(),
        }
    }

    fn run(mut self, rx: Receiver<Msg>) -> Result<CameraResult> {
        for msg in rx {
            match msg {
                Msg::Episode { index, length } => self.start_episode(index, length)?,
                Msg::Frame(frame) => self.frame(frame)?,
            }
        }
        self.end_episode()?;
        // ffmpeg 内部还缓存着未编码的帧，结束时需要等待
        self.bar.set_message("收尾");
        if let Some(ffmpeg) = self.ffmpeg.take() {
            ffmpeg.finish()?;
        }
        let (width, height) = self.size.context("没有收到任何帧")?;
        let info = video_info(&self.layout.path(0, 0), &self.encoder)?;
        crate::progress::finish_frame_bar(&self.bar);
        Ok(CameraResult { episodes: self.result_episodes, stats: self.result_stats, width, height, info })
    }

    fn start_episode(&mut self, index: usize, length: usize) -> Result<()> {
        self.end_episode()?;
        // lerobot：当前文件加上新 episode 超过大小限制时换新文件
        if let Some(ffmpeg) = &self.ffmpeg {
            let size_mb = fs::metadata(&ffmpeg.path).map(|m| m.len()).unwrap_or(0) as f64 / (1024.0 * 1024.0);
            if size_mb >= self.layout.file_size_mb {
                self.next_file()?;
            }
        }
        self.episode = Some(EpisodeState {
            index,
            length,
            received: 0,
            from_frame: self.frames_in_file,
            samples: sample_indices(length),
            next_sample: 0,
            pixel_counts: [[0; 256]; 3],
        });
        Ok(())
    }

    /// 结束当前视频文件，之后的帧写入下一个文件
    fn next_file(&mut self) -> Result<()> {
        if let Some(ffmpeg) = self.ffmpeg.take() {
            ffmpeg.finish()?;
        }
        if self.file_index + 1 == self.layout.chunks_size {
            self.chunk_index += 1;
            self.file_index = 0;
        } else {
            self.file_index += 1;
        }
        self.frames_in_file = 0;
        Ok(())
    }

    fn end_episode(&mut self) -> Result<()> {
        let Some(ep) = self.episode.take() else {
            return Ok(());
        };
        ensure!(ep.received == ep.length, "episode {} 应有 {} 帧，实际收到 {}", ep.index, ep.length, ep.received);
        let fps = self.layout.fps as f64;
        self.result_episodes.push(EpisodeVideo {
            episode_index: ep.index,
            chunk_index: self.chunk_index,
            file_index: self.file_index,
            from_timestamp: ep.from_frame as f64 / fps,
            to_timestamp: (ep.from_frame + ep.length) as f64 / fps,
        });
        self.result_stats
            .push(FeatureStats::from_pixel_counts(&ep.pixel_counts, ep.samples.len() as u64).scale(255.0));
        Ok(())
    }

    fn frame(&mut self, frame: Frame) -> Result<()> {
        let kind = frame.describe();
        // 不同输入文件的同一相机可能格式不同（如 JPEG 与 H.264），ffmpeg 的输入格式固定，只能换新文件；
        // 格式只会在输入文件之间变化，而文件边界一定是 episode 边界
        if self.ffmpeg.is_some() && self.ffmpeg_kind.as_deref() != Some(kind.as_str()) {
            let ep = self.episode.as_ref().context("收到帧之前没有开始 episode")?;
            ensure!(
                ep.received == 0,
                "episode {} 内图像格式变化：{} -> {kind}",
                ep.index,
                self.ffmpeg_kind.as_deref().unwrap_or("")
            );
            self.next_file()?;
            self.episode.as_mut().unwrap().from_frame = 0;
        }

        let ep = self.episode.as_mut().context("收到帧之前没有开始 episode")?;
        ensure!(ep.received < ep.length, "episode {} 收到的帧多于 {}", ep.index, ep.length);
        let frame_in_episode = ep.received;
        ep.received += 1;

        if ep.samples.get(ep.next_sample) == Some(&frame_in_episode) {
            // linspace 取整后可能重复，同一帧按出现次数计入
            let times = ep.samples[ep.next_sample..].iter().take_while(|&&i| i == frame_in_episode).count();
            ep.next_sample += times;
            let (w, h, rgb) = frame.decode_rgb()?;
            match self.size {
                None => self.size = Some((w, h)),
                Some(size) => ensure!(size == (w, h), "图像尺寸变化：{size:?} -> {:?}", (w, h)),
            }
            for _ in 0..times {
                accumulate_pixels(&mut ep.pixel_counts, w as usize, h as usize, &rgb);
            }
        }

        if self.ffmpeg.is_none() {
            let path = self.layout.path(self.chunk_index, self.file_index);
            self.ffmpeg = Some(Ffmpeg::spawn(&frame, &path, self.layout.fps, &self.encoder)?);
            self.ffmpeg_kind = Some(kind);
        }
        self.ffmpeg.as_mut().unwrap().write(&frame)?;
        self.frames_in_file += 1;
        self.bar.inc(1);
        Ok(())
    }
}

/// BT.601 limited range 的 YUV 4:2:0 转 RGB（HWC），与 ffmpeg 对未标注色彩空间的默认处理一致
fn yuv420p_to_rgb(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (y_plane, uv) = data.split_at(width * height);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let (u_plane, v_plane) = uv.split_at(cw * ch);
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let luma = 1.164 * (y_plane[y * width + x] as f32 - 16.0);
            let c = (y / 2) * cw + x / 2;
            let u = u_plane[c] as f32 - 128.0;
            let v = v_plane[c] as f32 - 128.0;
            for value in [luma + 1.596 * v, luma - 0.392 * u - 0.813 * v, luma + 2.017 * u] {
                rgb.push(value.round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    rgb
}

/// lerobot `sample_indices`：episode 内均匀抽样的帧号
fn sample_indices(len: usize) -> Vec<usize> {
    let num = if len < 100 { len } else { 100.max(((len as f64).powf(0.75) as usize).min(10_000)) };
    if num == 1 {
        return vec![0];
    }
    (0..num)
        .map(|i| {
            let x = i as f64 * (len - 1) as f64 / (num - 1) as f64;
            // np.round 是银行家舍入
            x.round_ties_even() as usize
        })
        .collect()
}

/// lerobot `auto_downsample_height_width` 后按通道累计像素值
fn accumulate_pixels(counts: &mut [[u64; 256]; 3], width: usize, height: usize, rgb: &[u8]) {
    let factor = if width.max(height) < 300 {
        1
    } else if width > height {
        width / 150
    } else {
        height / 150
    };
    for y in (0..height).step_by(factor) {
        for x in (0..width).step_by(factor) {
            let p = (y * width + x) * 3;
            for c in 0..3 {
                counts[c][rgb[p + c] as usize] += 1;
            }
        }
    }
}

/// 用 ffprobe 读取视频流信息，生成与 lerobot `get_video_info` 相同的字段
fn video_info(path: &Path, encoder: &VideoConfig) -> Result<Map<String, Json>> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate", "-of", "json"])
        .arg(path)
        .output()
        .context("启动 ffprobe 失败")?;
    ensure!(output.status.success(), "ffprobe {} 失败", path.display());
    let probe: Json = serde_json::from_slice(&output.stdout)?;
    let stream = &probe["streams"][0];
    let rate = stream["r_frame_rate"].as_str().context("ffprobe 缺少 r_frame_rate")?;
    let fps = match rate.split_once('/') {
        Some((n, d)) => n.parse::<f64>()? / d.parse::<f64>()?,
        None => rate.parse()?,
    };
    let pix_fmt = stream["pix_fmt"].as_str().context("ffprobe 缺少 pix_fmt")?;

    let mut info = Map::new();
    info.insert("video.height".into(), stream["height"].clone());
    info.insert("video.width".into(), stream["width"].clone());
    info.insert("video.codec".into(), stream["codec_name"].clone());
    info.insert("video.pix_fmt".into(), json!(pix_fmt));
    info.insert("video.fps".into(), json!(fps as i64));
    info.insert("video.channels".into(), json!(pix_fmt_channels(pix_fmt)));
    info.insert("has_audio".into(), json!(false));
    info.insert("video.g".into(), json!(encoder.g));
    info.insert("video.crf".into(), json!(encoder.crf));
    info.insert("video.preset".into(), json!(encoder.preset));
    info.insert("video.fast_decode".into(), json!(encoder.fast_decode));
    info.insert("video.video_backend".into(), json!("pyav"));
    info.insert("video.extra_options".into(), json!({}));
    info.insert("is_depth_map".into(), json!(false));
    Ok(info)
}

fn pix_fmt_channels(pix_fmt: &str) -> u32 {
    if pix_fmt.starts_with("gray") {
        1
    } else if pix_fmt.starts_with("yuva") || pix_fmt.starts_with("rgba") || pix_fmt.starts_with("bgra") {
        4
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::sample_indices;

    #[test]
    fn samples_like_lerobot() {
        assert_eq!(sample_indices(1), vec![0]);
        assert_eq!(sample_indices(5), vec![0, 1, 2, 3, 4]);
        let s = sample_indices(247);
        assert_eq!(s.len(), 100);
        assert_eq!((s[0], s[99]), (0, 246));
        // 1000 帧时 lerobot 抽 177 帧
        assert_eq!(sample_indices(1000).len(), 177);
    }
}
