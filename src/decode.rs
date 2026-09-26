//! 用 ffmpeg 子进程顺序解码 H.264 码流，得到 yuv420p 原始帧

use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::VecDeque,
    io::{Read, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        mpsc::{Receiver, channel},
    },
    thread::JoinHandle,
};

use crate::video::{Frame, RawFormat};

/// 一个 H.264 码流的解码器。每条消息（一个 access unit）解出一帧，输出顺序与输入顺序一致；
/// 调用方需保证码流从关键帧开始且不含 B 帧
pub struct StreamDecoder {
    child: Child,
    stdin: Option<ChildStdin>,
    frames: Receiver<Vec<u8>>,
    reader: Option<JoinHandle<Result<()>>>,
    width: u32,
    height: u32,
    /// 已送入、尚未取回帧的消息的附带数据（使用该帧的全局帧号）
    pending: VecDeque<Vec<usize>>,
    fed: usize,
    received: usize,
}

impl StreamDecoder {
    /// `first` 必须是包含 SPS/PPS 的关键帧，用于得到分辨率
    pub fn spawn(first: &[u8]) -> Result<Self> {
        let (width, height) = probe_size(first)?;
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "pipe:0"])
            // passthrough：不按帧率补帧或丢帧，保证一条消息对应一帧
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("启动 ffmpeg 解码失败")?;
        let stdin = child.stdin.take();
        let mut stdout = child.stdout.take().unwrap();
        let frame_size = (width * height * 3 / 2) as usize;
        let (tx, frames) = channel();
        let reader = std::thread::spawn(move || -> Result<()> {
            loop {
                let mut buf = vec![0u8; frame_size];
                match read_full(&mut stdout, &mut buf)? {
                    0 => return Ok(()),
                    n if n == frame_size => {
                        if tx.send(buf).is_err() {
                            return Ok(());
                        }
                    }
                    n => bail!("解码输出不完整：{n}/{frame_size} 字节"),
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            frames,
            reader: Some(reader),
            width,
            height,
            pending: VecDeque::new(),
            fed: 0,
            received: 0,
        })
    }

    /// 送入一条消息，`users` 为使用这一帧的全局帧号（可以为空）；返回已经解出的帧
    pub fn push(&mut self, data: &[u8], users: Vec<usize>) -> Result<Vec<(Vec<usize>, Frame)>> {
        let stdin = self.stdin.as_mut().expect("解码器已结束");
        if stdin.write_all(data).is_err() {
            bail!("ffmpeg 解码异常退出：{}", self.stderr());
        }
        self.pending.push_back(users);
        self.fed += 1;
        let mut out = Vec::new();
        while let Ok(buf) = self.frames.try_recv() {
            out.push(self.pair(buf)?);
        }
        Ok(out)
    }

    /// 结束输入并取回剩余的帧；解出的帧数必须与送入的消息数相同
    pub fn finish(mut self) -> Result<Vec<(Vec<usize>, Frame)>> {
        drop(self.stdin.take());
        let mut out = Vec::new();
        while let Ok(buf) = self.frames.recv() {
            out.push(self.pair(buf)?);
        }
        self.reader.take().unwrap().join().map_err(|_| anyhow::anyhow!("解码读取线程 panic"))??;
        let stderr = self.stderr();
        let status = self.child.wait()?;
        ensure!(status.success(), "ffmpeg 解码失败：{stderr}");
        ensure!(
            self.received == self.fed,
            "H.264 解码帧数 {} 与消息数 {} 不一致，无法可靠对应（码流可能有损坏）{}",
            self.received,
            self.fed,
            if stderr.is_empty() { String::new() } else { format!("：{stderr}") }
        );
        Ok(out)
    }

    fn pair(&mut self, buf: Vec<u8>) -> Result<(Vec<usize>, Frame)> {
        let users = self.pending.pop_front().context("解码出的帧多于送入的消息")?;
        self.received += 1;
        let frame = Frame::Raw { data: Arc::new(buf), width: self.width, height: self.height, format: RawFormat::Yuv420p };
        Ok((users, frame))
    }

    fn stderr(&mut self) -> String {
        let mut s = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            let _ = e.read_to_string(&mut s);
        }
        s.trim().to_string()
    }
}

/// 读满 buf，返回实际读到的字节数（EOF 时可能不足）
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

/// 用 ffprobe 从关键帧中读出分辨率
fn probe_size(keyframe: &[u8]) -> Result<(u32, u32)> {
    let mut child = Command::new("ffprobe")
        .args(["-v", "error", "-f", "h264", "-i", "pipe:0", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("启动 ffprobe 失败")?;
    // 写完后关闭 stdin，ffprobe 才会结束
    let _ = child.stdin.take().unwrap().write_all(keyframe);
    let output = child.wait_with_output()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let (w, h) = text.trim().split_once(',').context("无法从 H.264 关键帧读取分辨率")?;
    Ok((w.parse()?, h.parse()?))
}
