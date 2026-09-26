//! 进度条样式，以及按最近一秒计算的速度和剩余时间

use indicatif::{
    HumanCount, HumanDuration, MultiProgress, ProgressBar, ProgressState, ProgressStyle,
    style::ProgressTracker,
};
use std::{
    collections::VecDeque,
    fmt::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

const WINDOW: Duration = Duration::from_secs(1);

#[derive(Clone, Copy)]
enum Show {
    Rate,
    Remaining,
}

/// 记录最近一秒内的 (时间, 位置) 采样，用首尾两个采样计算速度
#[derive(Clone)]
struct WindowRate {
    show: Show,
    samples: VecDeque<(Instant, u64)>,
}

impl WindowRate {
    fn new(show: Show) -> Self {
        Self { show, samples: VecDeque::new() }
    }

    fn record(&mut self, now: Instant, pos: u64) {
        self.samples.push_back((now, pos));
        // 保留窗口起点之前的最后一个采样，使计算跨度覆盖完整的一秒
        while self.samples.len() > 2 && now - self.samples[1].0 >= WINDOW {
            self.samples.pop_front();
        }
    }

    /// 每秒步数；采样不足时为 None
    fn per_sec(&self) -> Option<f64> {
        let (&(t0, p0), &(t1, p1)) = (self.samples.front()?, self.samples.back()?);
        let secs = (t1 - t0).as_secs_f64();
        (secs > 0.0).then(|| p1.saturating_sub(p0) as f64 / secs)
    }
}

impl ProgressTracker for WindowRate {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, state: &ProgressState, now: Instant) {
        self.record(now, state.pos());
    }

    fn reset(&mut self, _: &ProgressState, _: Instant) {
        self.samples.clear();
    }

    fn write(&self, state: &ProgressState, w: &mut dyn Write) {
        let rate = self.per_sec();
        let _ = match self.show {
            Show::Rate => write!(w, "{}/s", HumanCount(rate.unwrap_or(0.0) as u64)),
            Show::Remaining => match (rate, state.len()) {
                (Some(r), Some(len)) if r > 0.0 => {
                    let secs = len.saturating_sub(state.pos()) as f64 / r;
                    write!(w, "{:#}", HumanDuration(Duration::from_secs_f64(secs)))
                }
                _ => write!(w, "-"),
            },
        };
    }
}

/// 按消息数计数的进度条；`{rate}`、`{remaining}` 基于最近一秒
pub fn message_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:>12} [{bar:40.cyan/blue}] {human_pos}/{human_len} 条消息  {rate}  剩余 {remaining}",
    )
    .unwrap()
    .with_key("rate", WindowRate::new(Show::Rate))
    .with_key("remaining", WindowRate::new(Show::Remaining))
    .progress_chars("=> ")
}

/// 总数未知时的计数器
pub fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:>12} {spinner} {human_pos} 条消息  {rate}")
        .unwrap()
        .with_key("rate", WindowRate::new(Show::Rate))
}

/// 单个 feature 已完成的帧数；`{rate}`、`{remaining}` 基于最近一秒
pub fn frame_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:>12} [{bar:40.green/white}] {pos}/{len} 帧  {rate}  剩余 {remaining}  {msg}",
    )
    .unwrap()
    .with_key("rate", WindowRate::new(Show::Rate))
    .with_key("remaining", WindowRate::new(Show::Remaining))
    .progress_chars("=> ")
}

/// feature 完成后显示总用时，替换掉不再有意义的速度和剩余时间
pub fn finish_frame_bar(bar: &ProgressBar) {
    bar.set_style(
        ProgressStyle::with_template("{prefix:>12} [{bar:40.green/white}] {pos}/{len} 帧  完成，用时 {msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    bar.finish_with_message(format!("{:#}", HumanDuration(bar.elapsed())));
}

/// 整个转换的总进度：定时汇总各 feature 进度条的帧数，直到显式结束
pub struct TotalProgress {
    bar: ProgressBar,
    stop: Arc<AtomicBool>,
    poller: Option<JoinHandle<()>>,
}

impl TotalProgress {
    /// 插在 `progress` 的最上方；`parts` 为各 feature 的进度条
    pub fn start(progress: &MultiProgress, parts: Vec<ProgressBar>) -> Self {
        let style = ProgressStyle::with_template(
            "{prefix:>12} [{bar:40.yellow/white}] {percent:>3}%  剩余 {remaining}  {msg}",
        )
        .unwrap()
        .with_key("remaining", WindowRate::new(Show::Remaining))
        .progress_chars("=> ");
        let bar = progress.insert(0, ProgressBar::no_length().with_style(style).with_prefix("总进度"));
        let stop = Arc::new(AtomicBool::new(false));
        let poller = {
            let (bar, stop) = (bar.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let len: u64 = parts.iter().map(|b| b.length().unwrap_or(0)).sum();
                    let pos = parts.iter().map(|b| b.position()).sum();
                    // 总数未知前保持空条，避免长度 0 显示为 100%
                    if len > 0 {
                        bar.set_length(len);
                    }
                    bar.set_position(pos);
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
        };
        Self { bar, stop, poller: Some(poller) }
    }

    /// 显示当前所处阶段
    pub fn set_stage(&self, stage: &'static str) {
        self.bar.set_message(stage);
    }

    fn stop_poller(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(poller) = self.poller.take() {
            let _ = poller.join();
        }
    }

    pub fn finish(mut self) {
        self.stop_poller();
        self.bar.set_style(
            ProgressStyle::with_template("{prefix:>12} [{bar:40.yellow/white}] 100%  完成，用时 {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        if let Some(len) = self.bar.length() {
            self.bar.set_position(len);
        }
        self.bar.finish_with_message(format!("{:#}", HumanDuration(self.bar.elapsed())));
    }
}

impl Drop for TotalProgress {
    /// 出错提前返回时也要停掉汇总线程
    fn drop(&mut self) {
        self.stop_poller();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_uses_last_second() {
        let mut r = WindowRate::new(Show::Rate);
        let t0 = Instant::now();
        let ms = |n| t0 + Duration::from_millis(n);
        r.record(t0, 0);
        assert_eq!(r.per_sec(), None);
        r.record(ms(500), 1000);
        r.record(ms(1500), 1100);
        // t=0 的采样已超出窗口，只用 0.5s 到 1.5s 这一秒
        assert_eq!(r.samples.len(), 2);
        assert!((r.per_sec().unwrap() - 100.0).abs() < 1e-9);
        // 停滞时速度降为 0
        r.record(ms(2600), 1100);
        assert_eq!(r.per_sec(), Some(0.0));
    }
}
