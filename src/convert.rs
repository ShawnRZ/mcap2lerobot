use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use indicatif::{HumanBytes, MultiProgress, ProgressBar};
use mcap::{MessageStream, Summary};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use crate::cdr::CdrReader;
use crate::config::{Config, FeatureKind};
use crate::dataset::{
    self, DataWriter, EpisodeMeta, FeatureSpec, Format, MetaWriter, VIDEO_FILE_SIZE_MB, VideoLocation,
};
use crate::inputs;
use crate::decode::StreamDecoder;
use crate::h264::{self, KeyframeGate};
use crate::map_file;
use crate::progress::{TotalProgress, finish_frame_bar, frame_style, message_style, spinner_style};
use crate::ros2msg::{self, Value};
use crate::stats::FeatureStats;
use crate::sync::{self, ConfigPlan};
use crate::video::{CameraEncoder, Frame, RawFormat, VideoLayout};

const COMPRESSED_IMAGE: &str = "sensor_msgs/msg/CompressedImage";
const RAW_IMAGE: &str = "sensor_msgs/msg/Image";

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// 输入的 MCAP 文件或文件夹，可指定多个；文件夹会递归查找其中的 .mcap 文件（按路径排序）。
    /// 切分出的 episode 按顺序追加
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// topic -> feature 映射配置文件（TOML）
    #[arg(short, long)]
    config: PathBuf,

    /// 输出目录
    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    /// 目标帧率，覆盖配置文件中的 fps
    #[arg(long)]
    fps: Option<u32>,

    /// 任务描述，覆盖配置文件中的 task
    #[arg(long)]
    task: Option<String>,

    /// 输出目录已存在时删除后重新生成（只会删除 LeRobot 数据集目录或空目录）
    #[arg(long)]
    force: bool,

    /// 输出的 LeRobot 数据集版本：v3.0（lerobot ≥ 0.4）或 v2.1（lerobot 0.1 ~ 0.3）
    #[arg(long, value_enum, default_value = "v3.0")]
    format: Format,

    /// 跳过开始转换前的确认
    #[arg(short, long)]
    yes: bool,
}

/// vector feature 的一个 source 在转换过程中的状态
struct SourceState {
    feature: usize,
    topic: String,
    field: String,
    configured_names: Vec<String>,
    /// 第一次解码后确定；可选 source 由 fill 预先确定，与文件中是否有数据无关
    names: Option<Vec<String>>,
    /// 可选 source 缺失时每帧使用的值
    fill: Option<Rc<Vec<f64>>>,
}

/// 按帧号顺序把帧送入编码器；消息在文件中的顺序与帧号不完全一致，先缓存再按序输出
struct CameraFeed {
    encoder: CameraEncoder,
    /// H.264 输入：当前文件的码流是否已到第一个关键帧，以及解码器
    gate: KeyframeGate,
    decoder: Option<StreamDecoder>,
    pending: BTreeMap<usize, Frame>,
    next: usize,
    /// 全局帧号 -> (episode_index, 长度)
    episode_starts: HashMap<usize, (usize, usize)>,
}

impl CameraFeed {
    fn insert(&mut self, frame_index: usize, frame: Frame) -> Result<()> {
        self.pending.insert(frame_index, frame);
        while let Some(frame) = self.pending.remove(&self.next) {
            if let Some(&(episode, length)) = self.episode_starts.get(&self.next) {
                self.encoder.start_episode(episode, length)?;
            }
            self.encoder.push(frame)?;
            self.next += 1;
        }
        Ok(())
    }

    /// 处理一条图像消息，`frames` 为使用它的全局帧号。H.264 消息即使没有帧使用也要送入解码器，
    /// 因为后续的 P 帧依赖它
    fn handle(&mut self, schema: &str, data: &[u8], frames: &[usize]) -> Result<()> {
        match image_payload(schema, data)? {
            ImagePayload::H264(bytes) => {
                // 第一个关键帧之前的消息无法解码；第一遍扫描已把它们排除，不会被任何帧使用
                if !self.gate.accept(bytes) {
                    return Ok(());
                }
                if self.decoder.is_none() {
                    self.decoder = Some(StreamDecoder::spawn(bytes)?);
                }
                let decoded = self.decoder.as_mut().unwrap().push(bytes, frames.to_vec())?;
                self.insert_decoded(decoded)
            }
            _ if frames.is_empty() => Ok(()),
            ImagePayload::Compressed(bytes) => {
                let frame = Frame::Encoded(Arc::new(bytes.to_vec()));
                frames.iter().try_for_each(|&g| self.insert(g, frame.clone()))
            }
            ImagePayload::Raw(frame) => frames.iter().try_for_each(|&g| self.insert(g, frame.clone())),
        }
    }

    fn insert_decoded(&mut self, decoded: Vec<(Vec<usize>, Frame)>) -> Result<()> {
        for (users, frame) in decoded {
            for g in users {
                self.insert(g, frame.clone())?;
            }
        }
        Ok(())
    }

    /// 输入文件结束：取回解码器中剩余的帧，下一个文件的码流重新从关键帧开始
    fn end_file(&mut self) -> Result<()> {
        self.gate = KeyframeGate::default();
        if let Some(decoder) = self.decoder.take() {
            let rest = decoder.finish()?;
            self.insert_decoded(rest)?;
        }
        Ok(())
    }
}

/// 转换过程中跨文件的全局状态
struct Converter<'a> {
    config: &'a Config,
    fps: u32,
    format: Format,
    task: String,
    output: PathBuf,
    sources: Vec<SourceState>,
    /// (feature 下标, topic, 编码器)
    cameras: Vec<(usize, String, CameraFeed)>,
    data: Option<DataWriter>,
    feature_specs: Option<Vec<FeatureSpec>>,
    episodes: Vec<EpisodeMeta>,
    total_frames: usize,
    progress: MultiProgress,
    /// 每个 feature 一个进度条，与 config.features 一一对应
    feature_bars: Vec<ProgressBar>,
    total: Option<TotalProgress>,
}

pub fn run(args: &ConvertArgs, _verbose: u8) -> Result<()> {
    let mut config = Config::load(&args.config)?;
    if let Some(fps) = args.fps {
        config.fps = fps;
    }
    let task = args
        .task
        .clone()
        .or_else(|| config.task.clone())
        .context("缺少任务描述：在配置中设置 task 或使用 --task")?;

    let inputs = inputs::collect(&args.inputs)?;
    let overwrite = check_output(&args.output, args.force)?;
    // 文件夹展开后的文件列表需要用户确认
    if inputs.has_dir {
        print_summary(&inputs.files, &args.output, overwrite)?;
        if !inputs::confirm("是否开始转换？", args.yes)? {
            println!("已取消");
            return Ok(());
        }
    }
    if overwrite {
        fs::remove_dir_all(&args.output).with_context(|| format!("删除 {} 失败", args.output.display()))?;
    }

    let files = &inputs.files;
    let mut converter = Converter::new(&config, args.format, task, &args.output);
    for (i, input) in files.iter().enumerate() {
        converter.log(format!("[{}/{}] {}", i + 1, files.len(), input.display()));
        converter.convert_file(input).with_context(|| format!("转换 {} 失败", input.display()))?;
    }
    converter.finish()
}

/// 检查输出目录，返回是否需要先删除已有目录
fn check_output(output: &Path, force: bool) -> Result<bool> {
    if !output.exists() {
        return Ok(false);
    }
    if !force {
        bail!("输出目录 {} 已存在，使用 --force 覆盖", output.display());
    }
    let is_dataset = output.join("meta/info.json").exists();
    let is_empty = fs::read_dir(output)?.next().is_none();
    ensure!(
        is_dataset || is_empty,
        "{} 不是 LeRobot 数据集目录（没有 meta/info.json）且不为空，拒绝删除",
        output.display()
    );
    Ok(true)
}

/// 列出待转换的文件、数量和总大小
fn print_summary(files: &[PathBuf], output: &Path, overwrite: bool) -> Result<()> {
    const MAX_LISTED: usize = 20;
    let sizes: Vec<u64> = files.iter().map(|f| fs::metadata(f).map(|m| m.len())).collect::<Result<_, _>>()?;
    println!("找到 {} 个 MCAP 文件，共 {}：", files.len(), HumanBytes(sizes.iter().sum()));
    for (f, size) in files.iter().zip(&sizes).take(MAX_LISTED) {
        println!("  {:>10}  {}", HumanBytes(*size).to_string(), f.display());
    }
    if files.len() > MAX_LISTED {
        println!("  ……其余 {} 个文件未列出", files.len() - MAX_LISTED);
    }
    let state = if overwrite { "（已存在，将被删除后重新生成）" } else { "" };
    println!("输出目录：{}{state}", output.display());
    Ok(())
}

/// 第一遍扫描得到的单个 topic 信息
#[derive(Default)]
struct TopicScan {
    /// 按文件顺序的 (stamp, 序号)
    stamps: Vec<i64>,
    missing_header: bool,
}

impl<'a> Converter<'a> {
    fn new(config: &'a Config, format: Format, task: String, output: &Path) -> Self {
        let progress = MultiProgress::new();
        let feature_bars: Vec<ProgressBar> = config
            .features
            .iter()
            .map(|f| {
                let short_name = f.name.rsplit('.').next().unwrap_or(&f.name).to_string();
                // 总帧数要等第一个文件对齐后才知道；不设长度时显示为空条（长度 0 会显示为满条）
                let bar = progress.add(ProgressBar::no_length().with_style(frame_style()).with_prefix(short_name));
                // 定时刷新，停滞时速度能降为 0
                bar.enable_steady_tick(std::time::Duration::from_millis(100));
                bar
            })
            .collect();
        let mut sources = Vec::new();
        let mut cameras = Vec::new();
        for (fi, f) in config.features.iter().enumerate() {
            match &f.kind {
                FeatureKind::Video { topic, .. } => {
                    let layout = VideoLayout {
                        root: output.to_path_buf(),
                        format,
                        video_key: f.name.clone(),
                        fps: config.fps,
                        chunks_size: dataset::CHUNKS_SIZE,
                        file_size_mb: VIDEO_FILE_SIZE_MB,
                    };
                    let feed = CameraFeed {
                        encoder: CameraEncoder::spawn(layout, config.video.clone(), feature_bars[fi].clone()),
                        gate: KeyframeGate::default(),
                        decoder: None,
                        pending: BTreeMap::new(),
                        next: 0,
                        episode_starts: HashMap::new(),
                    };
                    cameras.push((fi, topic.clone(), feed));
                }
                FeatureKind::Vector { sources: srcs } => {
                    for s in srcs {
                        let fill = s.fill.clone().filter(|_| s.optional);
                        // 可选 source 的维度和名字在配置中就能确定，避免随第一个文件是否有数据而变化
                        let names = fill.as_ref().map(|fill| {
                            if s.names.is_empty() {
                                let prefix = s.topic.rsplit('/').next().unwrap_or(&s.topic);
                                (0..fill.len()).map(|i| format!("{prefix}.{}[{i}]", s.field)).collect()
                            } else {
                                s.names.clone()
                            }
                        });
                        sources.push(SourceState {
                            feature: fi,
                            topic: s.topic.clone(),
                            field: s.field.clone(),
                            configured_names: s.names.clone(),
                            names,
                            fill: fill.map(Rc::new),
                        });
                    }
                }
            }
        }
        Self {
            config,
            fps: config.fps,
            format,
            task,
            output: output.to_path_buf(),
            sources,
            cameras,
            data: None,
            feature_specs: None,
            episodes: Vec::new(),
            total_frames: 0,
            total: Some(TotalProgress::start(&progress, feature_bars.clone())),
            progress,
            feature_bars,
        }
    }

    /// 输出一行文字；进度条隐藏（非终端）时直接打印
    fn log(&self, msg: impl AsRef<str>) {
        if self.progress.is_hidden() {
            println!("{}", msg.as_ref());
        } else {
            let _ = self.progress.println(msg.as_ref());
        }
    }

    /// 按消息数显示进度的进度条，插在相机进度条上方
    fn message_bar(&self, prefix: &str, total: Option<u64>) -> ProgressBar {
        let bar = match total {
            Some(n) => ProgressBar::new(n).with_style(message_style()),
            None => ProgressBar::new_spinner().with_style(spinner_style()),
        };
        // 位于总进度条下方、各 feature 进度条上方
        let bar = self.progress.insert(1, bar.with_prefix(prefix.to_string()));
        // 定时刷新：处理被视频编码阻塞时，速度能如实降下来
        bar.enable_steady_tick(std::time::Duration::from_millis(100));
        bar
    }

    fn convert_file(&mut self, input: &Path) -> Result<()> {
        let mapped = map_file(input)?;
        let total_messages = Summary::read(&mapped)?.and_then(|s| s.stats).map(|s| s.message_count);

        // ---------- 第一遍：收集 stamp，生成对齐方案 ----------
        let bar = self.message_bar("扫描", total_messages);
        let scans = scan_stamps(&mapped, self.config, &bar)?;
        bar.finish_and_clear();
        let ConfigPlan { topics, missing_optional, plan, .. } =
            sync::plan_for_config(self.config, |t| scans[t].stamps.clone())?;
        self.log(format!("  {} 个 episode，保留 {} 帧", plan.segments.len(), plan.kept_frames()));
        for topic in &missing_optional {
            self.log(format!("  可选 topic 缺失，用 fill 填充：{topic}"));
        }
        if plan.segments.is_empty() {
            return Ok(());
        }
        for bar in &self.feature_bars {
            // inc_length 对未设长度的进度条无效
            bar.set_length(bar.length().unwrap_or(0) + plan.kept_frames() as u64);
        }

        // 每个 topic：文件中第 n 条消息 -> 排序后的位置（stamp 为 0 的消息没有位置）
        let sorted_pos: HashMap<&str, Vec<Option<usize>>> = topics
            .iter()
            .map(|(topic, _)| {
                let stamps = &scans[*topic].stamps;
                let mut order: Vec<usize> = (0..stamps.len()).filter(|&i| stamps[i] != 0).collect();
                // 与 sync 中的排序一致：按 stamp 排序，相同 stamp 保持文件顺序
                order.sort_by_key(|&i| stamps[i]);
                let mut pos = vec![None; stamps.len()];
                for (p, &i) in order.iter().enumerate() {
                    pos[i] = Some(p);
                }
                (*topic, pos)
            })
            .collect();

        // 每个 topic：排序后的位置 -> 使用它的全局帧号
        let frame_base = self.total_frames;
        let mut users: HashMap<&str, HashMap<usize, Vec<usize>>> = HashMap::new();
        let mut global = frame_base;
        let mut episode_ranges = Vec::new();
        for seg in &plan.segments {
            let episode_index = self.episodes.len() + episode_ranges.len();
            for (_, _, feed) in &mut self.cameras {
                feed.episode_starts.insert(global, (episode_index, seg.len()));
            }
            episode_ranges.push((episode_index, global, seg.len()));
            for frame in seg.clone() {
                for ((topic, _), m) in topics.iter().zip(&plan.topics) {
                    users.entry(topic).or_default().entry(m.matched[frame]).or_default().push(global);
                }
                global += 1;
            }
        }
        let kept = global - frame_base;

        // ---------- 第二遍：分发图像和数值 ----------
        let mut ordinals: HashMap<&str, usize> = HashMap::new();
        let mut schemas: HashMap<u16, Rc<ros2msg::Schema>> = HashMap::new();
        let mut rows: Vec<Vec<Option<Rc<Vec<f64>>>>> = vec![vec![None; kept]; self.sources.len()];
        // vector feature 每帧已到齐的 source 数，全部到齐时该 feature 的进度加一
        let mut sources_per_feature = vec![0usize; self.config.features.len()];
        for s in &self.sources {
            sources_per_feature[s.feature] += 1;
        }
        let mut filled = vec![vec![0usize; kept]; self.config.features.len()];
        // 缺失的可选 source：所有帧直接填充
        for (si, source) in self.sources.iter().enumerate() {
            if !missing_optional.contains(&source.topic.as_str()) {
                continue;
            }
            let fill = source.fill.clone().expect("缺失的可选 topic 必有 fill");
            for local in 0..kept {
                rows[si][local] = Some(fill.clone());
                filled[source.feature][local] += 1;
                if filled[source.feature][local] == sources_per_feature[source.feature] {
                    self.feature_bars[source.feature].inc(1);
                }
            }
        }

        let bar = self.message_bar("转换", total_messages);
        for msg in MessageStream::new(&mapped)? {
            let msg = msg?;
            bar.inc(1);
            let topic = msg.channel.topic.as_str();
            let Some((&key, positions)) = sorted_pos.get_key_value(topic) else {
                continue;
            };
            let ordinal = ordinals.entry(key).or_default();
            let pos = positions[*ordinal];
            *ordinal += 1;
            let frames: &[usize] = pos.and_then(|p| users[topic].get(&p)).map_or(&[], |v| v.as_slice());
            let schema_name = msg.channel.schema.as_ref().map(|s| s.name.as_str()).unwrap_or("");

            for (_, cam_topic, feed) in &mut self.cameras {
                if cam_topic == topic {
                    feed.handle(schema_name, &msg.data, frames)
                        .with_context(|| format!("处理 {topic} 的图像失败"))?;
                }
            }
            if frames.is_empty() {
                continue;
            }

            if self.sources.iter().any(|s| s.topic == topic) {
                let schema = match schemas.get(&msg.channel.id) {
                    Some(s) => s.clone(),
                    None => {
                        let s = msg.channel.schema.as_ref().context("channel 没有 schema")?;
                        ensure!(s.encoding == "ros2msg", "{topic} 的 schema 编码 {} 不是 ros2msg", s.encoding);
                        let text = std::str::from_utf8(&s.data).context("schema 不是 UTF-8")?;
                        let parsed = Rc::new(ros2msg::Schema::parse(&s.name, text)?);
                        schemas.insert(msg.channel.id, parsed.clone());
                        parsed
                    }
                };
                ensure!(msg.channel.message_encoding == "cdr", "{topic} 的消息编码不是 cdr");
                let value = schema.decode(&msg.data).with_context(|| format!("解码 {topic} 失败"))?;
                for (si, source) in self.sources.iter_mut().enumerate() {
                    if source.topic != topic {
                        continue;
                    }
                    let values = Rc::new(source.extract(&value)?);
                    for &g in frames {
                        rows[si][g - frame_base] = Some(values.clone());
                        let count = &mut filled[source.feature][g - frame_base];
                        *count += 1;
                        if *count == sources_per_feature[source.feature] {
                            self.feature_bars[source.feature].inc(1);
                        }
                    }
                }
            }
        }

        bar.finish_and_clear();
        for (_, topic, feed) in &mut self.cameras {
            feed.end_file().with_context(|| format!("解码 {topic} 失败"))?;
            ensure!(feed.next == global, "{topic} 缺少帧 {}", feed.next);
        }

        // ---------- 写出各 episode 的数值数据 ----------
        let specs = self.feature_specs()?;
        let data = self.data.get_or_insert_with(|| DataWriter::new(&self.output, self.format, self.fps, &specs));
        for (episode_index, start, length) in episode_ranges {
            let mut vectors = Vec::new();
            let mut stats = HashMap::new();
            for spec in &specs {
                let FeatureSpec::Vector { name, names } = spec else {
                    continue;
                };
                let fi = self.config.features.iter().position(|f| &f.name == name).unwrap();
                let mut values: Vec<f32> = Vec::with_capacity(length * names.len());
                for g in start..start + length {
                    for (si, source) in self.sources.iter().enumerate() {
                        if source.feature == fi {
                            let row = rows[si][g - frame_base].as_ref().context("缺少数值数据")?;
                            values.extend(row.iter().map(|&v| v as f32));
                        }
                    }
                }
                let as_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
                stats.insert(name.clone(), FeatureStats::compute(&as_f64, names.len(), length as u64, false));
                vectors.push(values);
            }
            let location = data.write_episode(episode_index, 0, length, &vectors)?;
            for (name, s) in
                dataset::default_feature_stats(self.fps, episode_index, 0, location.from_index, length)
            {
                stats.insert(name.to_string(), s);
            }
            self.episodes.push(EpisodeMeta {
                episode_index,
                tasks: vec![self.task.clone()],
                length,
                data: location,
                videos: Vec::new(),
                stats,
            });
        }
        self.total_frames = global;
        Ok(())
    }

    /// 由第一次解码得到的维度生成 feature 列表
    fn feature_specs(&mut self) -> Result<Vec<FeatureSpec>> {
        if let Some(specs) = &self.feature_specs {
            return Ok(specs.clone());
        }
        let specs: Vec<FeatureSpec> = self
            .config
            .features
            .iter()
            .enumerate()
            .map(|(fi, f)| match &f.kind {
                FeatureKind::Video { .. } => Ok(FeatureSpec::Video { name: f.name.clone() }),
                FeatureKind::Vector { .. } => {
                    let mut names = Vec::new();
                    for s in self.sources.iter().filter(|s| s.feature == fi) {
                        names.extend(s.names.clone().context("source 尚未解码")?);
                    }
                    let unique: std::collections::HashSet<&String> = names.iter().collect();
                    if unique.len() != names.len() {
                        self.log(format!("警告：{} 的维度名有重复，可在配置的 source 中用 names 指定", f.name));
                    }
                    Ok(FeatureSpec::Vector { name: f.name.clone(), names })
                }
            })
            .collect::<Result<_>>()?;
        self.feature_specs = Some(specs.clone());
        Ok(specs)
    }

    fn finish(mut self) -> Result<()> {
        // vector feature 在最后一个文件的第二遍扫描结束时已全部完成
        for (f, bar) in self.config.features.iter().zip(&self.feature_bars) {
            if matches!(f.kind, FeatureKind::Vector { .. }) {
                finish_frame_bar(bar);
            }
        }
        let specs = match self.feature_specs.take() {
            Some(specs) => specs,
            None => bail!("没有生成任何 episode"),
        };
        if let Some(data) = self.data.take() {
            data.finish()?;
        }

        let total = self.total.take().expect("总进度只会结束一次");
        total.set_stage("等待视频编码完成");
        let mut video_info = HashMap::new();
        for (fi, _, feed) in std::mem::take(&mut self.cameras) {
            let name = self.config.features[fi].name.clone();
            let result = feed.encoder.finish()?;
            ensure!(result.episodes.len() == self.episodes.len(), "{name} 的 episode 数不一致");
            for ((ep, v), s) in self.episodes.iter_mut().zip(result.episodes).zip(result.stats) {
                ensure!(ep.episode_index == v.episode_index, "{name} 的 episode 顺序不一致");
                ep.videos.push(VideoLocation {
                    chunk_index: v.chunk_index,
                    file_index: v.file_index,
                    from_timestamp: v.from_timestamp,
                    to_timestamp: v.to_timestamp,
                });
                ep.stats.insert(name.clone(), s);
            }
            video_info.insert(name, (result.height, result.width, result.info));
        }

        let meta = MetaWriter {
            root: &self.output,
            format: self.format,
            fps: self.fps,
            robot_type: self.config.robot_type.as_deref(),
            features: &specs,
            video_info: &video_info,
            tasks: std::slice::from_ref(&self.task),
        };
        total.set_stage("写入元数据");
        meta.write(&self.episodes)?;
        total.finish();
        self.log(format!(
            "完成：{} 个 episode，{} 帧 -> {}",
            self.episodes.len(),
            self.total_frames,
            self.output.display()
        ));
        Ok(())
    }
}

impl SourceState {
    /// 从消息中取出 field 对应的数值；第一次调用时确定维度和名字，之后检查维度一致
    fn extract(&mut self, message: &Value) -> Result<Vec<f64>> {
        let target = message.get_path(&self.field)?;
        let mut values = Vec::new();
        let mut generated = Vec::new();
        target.flatten_numbers(&self.field, &mut values, &mut generated)?;

        match &self.names {
            Some(_) if self.fill.as_ref().is_some_and(|f| f.len() != values.len()) => bail!(
                "{} 的 {} 有 {} 维，与配置的 fill（{} 维）不一致",
                self.topic,
                self.field,
                values.len(),
                self.fill.as_ref().unwrap().len()
            ),
            Some(names) => ensure!(
                names.len() == values.len(),
                "{} 的 {} 维度变化：{} -> {}",
                self.topic,
                self.field,
                names.len(),
                values.len()
            ),
            None => self.names = Some(self.infer_names(message, target, generated, values.len())?),
        }
        Ok(values)
    }

    fn infer_names(&self, message: &Value, target: &Value, generated: Vec<String>, dim: usize) -> Result<Vec<String>> {
        if !self.configured_names.is_empty() {
            ensure!(
                self.configured_names.len() == dim,
                "{} 的 {} 有 {dim} 维，但配置了 {} 个 names",
                self.topic,
                self.field,
                self.configured_names.len()
            );
            return Ok(self.configured_names.clone());
        }
        // JointState 这类消息：顶层数值数组与同级的 name 字段对应
        if !self.field.contains('.')
            && matches!(target, Value::Array(_))
            && let Ok(name_field) = message.get_path("name")
            && let Some(names) = name_field.as_string_array()
            && names.iter().all(|n| !n.is_empty())
        {
            if names.len() == dim {
                return Ok(names.iter().map(|n| n.to_string()).collect());
            }
            // 只有一个名字对应多个值，如 ["left_arm"] 对应 6 个关节
            if names.len() == 1 {
                return Ok((0..dim).map(|i| format!("{}_{i}", names[0])).collect());
            }
        }
        // 用 topic 最后一段作前缀，避免不同 topic 的同名字段重名
        let prefix = self.topic.rsplit('/').next().unwrap_or(&self.topic);
        Ok(generated.into_iter().map(|g| format!("{prefix}.{g}")).collect())
    }
}

/// 第一遍扫描：读取配置中各 topic 所有消息的 header.stamp
fn scan_stamps(mapped: &[u8], config: &Config, bar: &ProgressBar) -> Result<HashMap<String, TopicScan>> {
    let mut scans: HashMap<String, TopicScan> =
        config.topics().map(|t| (t.to_string(), TopicScan::default())).collect();
    let mut header_cache: HashMap<u16, bool> = HashMap::new();
    let video_topics: HashSet<&str> = config
        .features
        .iter()
        .filter_map(|f| match &f.kind {
            FeatureKind::Video { topic, .. } => Some(topic.as_str()),
            FeatureKind::Vector { .. } => None,
        })
        .collect();
    let mut gates: HashMap<String, KeyframeGate> = HashMap::new();
    for msg in MessageStream::new(mapped)? {
        let msg = msg?;
        bar.inc(1);
        let topic = msg.channel.topic.as_str();
        let Some(scan) = scans.get_mut(topic) else {
            continue;
        };
        let has_header = *header_cache
            .entry(msg.channel.id)
            .or_insert_with(|| msg.channel.schema.as_deref().is_some_and(ros2msg::has_header));
        if !has_header || msg.channel.message_encoding != "cdr" {
            scan.missing_header = true;
            scan.stamps.push(0);
            continue;
        }
        let mut stamp = CdrReader::new(&msg.data)
            .and_then(|mut r| r.read_header())
            .with_context(|| format!("解析 {topic} 的 header 失败"))?;
        if video_topics.contains(topic) {
            let schema = msg.channel.schema.as_ref().map(|s| s.name.as_str()).unwrap_or("");
            if let ImagePayload::H264(bytes) =
                image_payload(schema, &msg.data).with_context(|| format!("解析 {topic} 的图像失败"))?
            {
                ensure!(!h264::has_b_slice(bytes), "{topic} 的 H.264 码流含 B 帧，暂不支持");
                // 第一个关键帧之前的消息无法解码，stamp 记为 0 使其不参与对齐
                if !gates.entry(topic.to_string()).or_default().accept(bytes) {
                    stamp = 0;
                }
            }
        }
        scan.stamps.push(stamp);
    }
    for (topic, scan) in &scans {
        // 可选 topic 允许缺失，是否缺失由 plan_for_config 判断
        ensure!(!scan.stamps.is_empty() || config.is_optional_topic(topic), "{topic} 没有任何消息");
        ensure!(!scan.missing_header, "{topic} 的消息没有 header.stamp（可先运行 verify 检查）");
    }
    Ok(scans)
}

/// 图像消息的内容
enum ImagePayload<'a> {
    /// JPEG / PNG，每条消息可独立解码
    Compressed(&'a [u8]),
    /// H.264 Annex-B 的一个 access unit，需按顺序解码
    H264(&'a [u8]),
    Raw(Frame),
}

/// 解析图像消息；CompressedImage 按 format 字段区分图片和视频码流
fn image_payload<'a>(schema: &str, data: &'a [u8]) -> Result<ImagePayload<'a>> {
    let mut r = CdrReader::new(data)?;
    r.read_header()?;
    match schema {
        COMPRESSED_IMAGE => {
            let format = r.read_string()?.to_ascii_lowercase();
            let n = r.read_u32()? as usize;
            let bytes = r.read_bytes(n)?;
            if format.contains("h264") {
                Ok(ImagePayload::H264(bytes))
            } else if format.contains("h265") || format.contains("hevc") {
                bail!("暂不支持 H.265 码流（format = {format}）")
            } else {
                Ok(ImagePayload::Compressed(bytes))
            }
        }
        RAW_IMAGE => {
            let height = r.read_u32()?;
            let width = r.read_u32()?;
            let format = RawFormat::from_encoding(r.read_string()?)?;
            r.read_u8()?; // is_bigendian
            let step = r.read_u32()? as usize;
            let n = r.read_u32()? as usize;
            let bytes = r.read_bytes(n)?;
            let row = width as usize * format.channels();
            ensure!(step >= row && n >= step * height as usize, "图像数据长度与尺寸不符");
            // 去掉行尾填充
            let tight: Vec<u8> =
                (0..height as usize).flat_map(|y| &bytes[y * step..y * step + row]).copied().collect();
            Ok(ImagePayload::Raw(Frame::Raw { data: Arc::new(tight), width, height, format }))
        }
        other => bail!("{other} 不是图像消息"),
    }
}
