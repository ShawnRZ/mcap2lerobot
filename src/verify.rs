use anyhow::{Result, bail};
use clap::Args;
use mcap::MessageStream;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
};

use crate::cdr::CdrReader;
use crate::config::{Config, FeatureKind};
use crate::inputs;
use crate::h264;
use crate::map_file;
use crate::ros2msg;
use crate::sync::{self, ConfigPlan};

/// 向报告缓冲区写一行
macro_rules! out {
    ($out:expr) => {
        $out.push('\n')
    };
    ($out:expr, $($arg:tt)*) => {{
        use std::fmt::Write as _;
        let _ = writeln!($out, $($arg)*);
    }};
}

const COMPRESSED_IMAGE: &str = "sensor_msgs/msg/CompressedImage";
const RAW_IMAGE: &str = "sensor_msgs/msg/Image";

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// 输入的 MCAP 文件或文件夹，可指定多个；文件夹会递归查找其中的 .mcap 文件。
    /// 多个文件时每个文件只输出一行结论，-v 输出完整报告，-vv 再加上坏帧明细
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// topic -> feature 映射配置文件（TOML）
    #[arg(short, long)]
    config: PathBuf,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TopicKind {
    Video,
    Vector,
}

/// 单个 topic 在扫描过程中收集到的信息
struct TopicReport<'a> {
    topic: &'a str,
    kind: TopicKind,
    features: Vec<&'a str>,
    schemas: BTreeSet<String>,
    message_encodings: BTreeSet<String>,
    /// schema 的第一个字段是否为 std_msgs/Header
    has_header: bool,
    message_count: usize,
    /// header.stamp（ns），按消息在文件中的顺序
    stamps: Vec<i64>,
    /// 图像编码 -> 消息数，仅 video topic
    image_encodings: BTreeMap<String, u64>,
    decode_errors: u64,
    first_decode_error: Option<String>,
    /// 所有引用都是 optional 的 source，允许缺失
    optional: bool,
    /// H.264 码流：是否已遇到第一个关键帧、关键帧前被排除的消息数、关键帧数、含 B 帧的消息数
    h264_started: bool,
    h264_pre_keyframe: usize,
    h264_keyframes: usize,
    h264_b_frames: usize,
}

impl<'a> TopicReport<'a> {
    fn new(topic: &'a str, kind: TopicKind) -> Self {
        Self {
            topic,
            kind,
            features: Vec::new(),
            schemas: BTreeSet::new(),
            message_encodings: BTreeSet::new(),
            has_header: false,
            message_count: 0,
            stamps: Vec::new(),
            image_encodings: BTreeMap::new(),
            decode_errors: 0,
            first_decode_error: None,
            optional: false,
            h264_started: false,
            h264_pre_keyframe: 0,
            h264_keyframes: 0,
            h264_b_frames: 0,
        }
    }
}

/// 一组数值的统计，单位与输入一致
struct Stats {
    mean: f64,
    max: f64,
    min: f64,
    variance: f64,
}

impl Stats {
    fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Some(Self {
            mean,
            max: values.iter().copied().fold(f64::MIN, f64::max),
            min: values.iter().copied().fold(f64::MAX, f64::min),
            variance,
        })
    }

    fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "平均 {:.3}  最大 {:.3}  最小 {:.3}  方差 {:.3} ms²  标准差 {:.3}",
            self.mean,
            self.max,
            self.min,
            self.variance,
            self.std_dev()
        )
    }
}

/// 相邻时间的间隔（ms），输入需已排序
fn intervals_ms(sorted: &[i64]) -> Vec<f64> {
    sorted.windows(2).map(|w| (w[1] - w[0]) as f64 / 1e6).collect()
}

/// 单个文件的检查结果
struct FileResult {
    /// 完整报告文本
    text: String,
    topics: usize,
    errors: usize,
    warnings: usize,
    episodes: usize,
    kept_frames: usize,
    reference_frames: usize,
}

pub fn run(args: &VerifyArgs, verbose: u8) -> Result<()> {
    let config = Config::load(&args.config)?;
    let files = inputs::collect(&args.inputs)?.files;
    if let [file] = files.as_slice() {
        // 单个文件：输出完整报告，-v 加上坏帧明细
        let result = verify_file(file, &config, verbose >= 1)?;
        print!("{}", result.text);
        println!();
        if result.errors > 0 {
            bail!("verify 失败：共 {} 个问题", result.errors);
        }
        println!("verify 通过：{} 个 topic 全部正常", result.topics);
        return Ok(());
    }

    let mut failed: Vec<(&PathBuf, String)> = Vec::new();
    let (mut episodes, mut kept, mut reference) = (0, 0, 0);
    for (i, file) in files.iter().enumerate() {
        let prefix = format!("[{}/{}]", i + 1, files.len());
        match verify_file(file, &config, verbose >= 2) {
            Ok(r) => {
                if verbose >= 1 {
                    println!("\n========== {prefix} {} ==========", file.display());
                    print!("{}", r.text);
                }
                episodes += r.episodes;
                kept += r.kept_frames;
                reference += r.reference_frames;
                let mark = if r.errors > 0 { "✗" } else { "✓" };
                let problems = if r.errors > 0 { format!("，{} 个问题", r.errors) } else { String::new() };
                let warnings = if r.warnings > 0 { format!("，{} 个警告", r.warnings) } else { String::new() };
                println!(
                    "{prefix} {mark} {}  {} 个 episode，保留 {}/{} 帧{problems}{warnings}",
                    file.display(),
                    r.episodes,
                    r.kept_frames,
                    r.reference_frames
                );
                if r.errors > 0 {
                    failed.push((file, format!("{} 个问题", r.errors)));
                }
            }
            Err(e) => {
                println!("{prefix} ✗ {}  {e:#}", file.display());
                failed.push((file, format!("{e:#}")));
            }
        }
    }

    println!(
        "\n汇总：{} 个文件，{} 个通过，{} 个失败；共 {episodes} 个 episode，保留 {kept}/{reference} 帧",
        files.len(),
        files.len() - failed.len(),
        failed.len()
    );
    if !failed.is_empty() {
        println!("失败的文件：");
        for (file, reason) in &failed {
            println!("  {}: {reason}", file.display());
        }
        bail!("verify 失败：{} 个文件有问题", failed.len());
    }
    Ok(())
}

/// 检查单个文件，报告写入返回值的 text；`bad_frame_detail` 时列出每个坏帧
fn verify_file(path: &Path, config: &Config, bad_frame_detail: bool) -> Result<FileResult> {
    let mut reports = collect_topics(config);
    let index: HashMap<&str, usize> =
        reports.iter().enumerate().map(|(i, r)| (r.topic, i)).collect();
    // channel id -> schema 是否带 Header
    let mut header_cache: HashMap<u16, bool> = HashMap::new();

    let mapped = map_file(path)?;
    for msg in MessageStream::new(&mapped)? {
        let msg = msg?;
        let Some(&i) = index.get(msg.channel.topic.as_str()) else {
            continue;
        };
        let report = &mut reports[i];
        let schema = msg.channel.schema.as_deref();
        let schema_name = schema.map(|s| s.name.as_str()).unwrap_or("");
        let has_header = *header_cache
            .entry(msg.channel.id)
            .or_insert_with(|| schema.is_some_and(ros2msg::has_header));

        report.schemas.insert(schema_name.to_string());
        report.message_encodings.insert(msg.channel.message_encoding.clone());
        report.has_header |= has_header;
        report.message_count += 1;

        if has_header && msg.channel.message_encoding == "cdr" {
            match decode(schema_name, report.kind, &msg.data) {
                Ok((stamp, encoding, h264)) => {
                    if let Some(enc) = encoding {
                        *report.image_encodings.entry(enc).or_default() += 1;
                    }
                    match h264 {
                        Some(H264Info { keyframe, b_slice }) => {
                            report.h264_keyframes += keyframe as usize;
                            report.h264_b_frames += b_slice as usize;
                            report.h264_started |= keyframe;
                            // 第一个关键帧之前的消息无法解码，与 convert 一致地排除
                            if report.h264_started {
                                report.stamps.push(stamp);
                            } else {
                                report.h264_pre_keyframe += 1;
                            }
                        }
                        None => report.stamps.push(stamp),
                    }
                }
                Err(e) => {
                    report.decode_errors += 1;
                    report.first_decode_error.get_or_insert_with(|| format!("{e:#}"));
                }
            }
        }
    }

    let mut out = String::new();
    let (mut errors, mut warnings) = (0, 0);
    for report in &reports {
        let (e, w) = print_report(&mut out, report, config.fps);
        errors += e;
        warnings += w;
    }
    let sync = check_sync(&mut out, &reports, config, bad_frame_detail);
    Ok(FileResult {
        text: out,
        topics: reports.len(),
        errors: errors + sync.errors,
        warnings: warnings + sync.warnings,
        episodes: sync.episodes,
        kept_frames: sync.kept_frames,
        reference_frames: sync.reference_frames,
    })
}

/// 按配置中首次出现的顺序收集 topic；同一 topic 被多个 feature 引用时，只要有一个是 video 就按 video 检查
fn collect_topics(config: &Config) -> Vec<TopicReport<'_>> {
    let mut reports: Vec<TopicReport> = Vec::new();
    for feature in &config.features {
        let entries: Vec<(&str, TopicKind)> = match &feature.kind {
            FeatureKind::Video { topic, .. } => vec![(topic, TopicKind::Video)],
            FeatureKind::Vector { sources } => {
                sources.iter().map(|s| (s.topic.as_str(), TopicKind::Vector)).collect()
            }
        };
        for (topic, kind) in entries {
            let report = match reports.iter_mut().find(|r| r.topic == topic) {
                Some(r) => r,
                None => reports.push_mut(TopicReport::new(topic, kind)),
            };
            if kind == TopicKind::Video {
                report.kind = TopicKind::Video;
            }
            if !report.features.contains(&feature.name.as_str()) {
                report.features.push(&feature.name);
            }
        }
    }
    for report in &mut reports {
        report.optional = config.is_optional_topic(report.topic);
    }
    reports
}

/// 解析带 Header 的 CDR 消息，返回 stamp（ns）；video topic 同时返回图像编码
struct H264Info {
    keyframe: bool,
    b_slice: bool,
}

/// 返回 stamp，video topic 另外返回图像编码；H.264 码流再返回帧类型信息
fn decode(schema: &str, kind: TopicKind, data: &[u8]) -> Result<(i64, Option<String>, Option<H264Info>)> {
    let mut r = CdrReader::new(data)?;
    let stamp = r.read_header()?;
    if kind != TopicKind::Video {
        return Ok((stamp, None, None));
    }
    let encoding = match schema {
        COMPRESSED_IMAGE => {
            let format = r.read_string()?.to_string();
            if format.to_ascii_lowercase().contains("h264") {
                let n = r.read_u32()? as usize;
                let bytes = r.read_bytes(n)?;
                let info = H264Info { keyframe: h264::is_keyframe(bytes), b_slice: h264::has_b_slice(bytes) };
                return Ok((stamp, Some(format), Some(info)));
            }
            Some(format)
        }
        RAW_IMAGE => {
            let height = r.read_u32()?;
            let width = r.read_u32()?;
            let encoding = r.read_string()?;
            Some(format!("{encoding} {width}x{height}"))
        }
        // 非图像 schema，由 print_report 统一报错
        _ => None,
    };
    Ok((stamp, encoding, None))
}

/// 输出单个 topic 的检查结果，返回 (问题数, 警告数)
fn print_report(out: &mut String, report: &TopicReport, fps: u32) -> (usize, usize) {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let kind = match report.kind {
        TopicKind::Video => "video",
        TopicKind::Vector => "vector",
    };
    out!(out, "\n[{kind}] {}", report.topic);
    out!(out, "  feature      : {}", report.features.join(", "));

    // 可选 topic：没有消息，或有 Header 但没有一个有效 stamp，视为缺失
    let no_valid_stamp = report.has_header && report.stamps.iter().all(|&st| st == 0);
    if report.optional && (report.message_count == 0 || no_valid_stamp) {
        out!(out, "  ! 可选 topic 缺失（{} 条消息），转换时用 fill 填充", report.message_count);
        return (0, 1);
    }
    if report.message_count == 0 {
        out!(out, "  ✗ 没有任何消息（topic 不存在或为空）");
        return (1, 0);
    }

    out!(out, "  schema       : {}", join(&report.schemas));
    out!(out, "  消息编码     : {}", join(&report.message_encodings));
    if report.schemas.len() > 1 {
        errors.push("存在多个 schema".into());
    }

    if report.kind == TopicKind::Video {
        if !report.schemas.iter().all(|s| s == COMPRESSED_IMAGE || s == RAW_IMAGE) {
            errors.push(format!("schema 不是 {COMPRESSED_IMAGE} 或 {RAW_IMAGE}"));
        }
        if !report.message_encodings.iter().all(|e| e == "cdr") {
            errors.push("消息编码不是 cdr，无法解析图像".into());
        }
        let encodings: Vec<String> =
            report.image_encodings.iter().map(|(enc, n)| format!("{enc} ({n})")).collect();
        let encodings = if encodings.is_empty() { "-".to_string() } else { encodings.join(", ") };
        out!(out, "  图像编码     : {encodings}");
        if report.image_encodings.len() > 1 {
            errors.push("图像编码不一致".into());
        }
        if report.image_encodings.keys().any(|e| {
            let e = e.to_ascii_lowercase();
            e.contains("h265") || e.contains("hevc")
        }) {
            errors.push("暂不支持 H.265 码流".into());
        }
        if report.h264_keyframes > 0 || report.h264_pre_keyframe > 0 {
            out!(
                out,
                "  H.264        : 关键帧 {}，含 B 帧的消息 {}，首个关键帧前 {} 条",
                report.h264_keyframes,
                report.h264_b_frames,
                report.h264_pre_keyframe
            );
            if report.h264_keyframes == 0 {
                errors.push("H.264 码流中没有关键帧，无法解码".into());
            }
            if report.h264_b_frames > 0 {
                errors.push("H.264 码流含 B 帧，暂不支持".into());
            }
            if report.h264_pre_keyframe > 0 && report.h264_keyframes > 0 {
                warnings.push(format!("{} 条消息在第一个关键帧之前，无法解码，已排除", report.h264_pre_keyframe));
            }
        }
    }
    if report.decode_errors > 0 {
        errors.push(format!(
            "{} 条消息解析失败，首个错误: {}",
            report.decode_errors,
            report.first_decode_error.as_deref().unwrap_or("")
        ));
    }

    out!(out, "  消息数       : {}", report.message_count);
    if report.has_header {
        check_stamps(out, &report.stamps, fps, &mut errors, &mut warnings);
    } else {
        errors.push("schema 没有 Header，无法获取 header.stamp".into());
    }

    for w in &warnings {
        out!(out, "  ! {w}");
    }
    for e in &errors {
        out!(out, "  ✗ {e}");
    }
    if warnings.is_empty() && errors.is_empty() {
        out!(out, "  ✓ OK");
    }
    (errors.len(), warnings.len())
}

/// 检查 header.stamp：0 值、单调性、间隔和频率；stamps 按消息在文件中的顺序排列
fn check_stamps(
    out: &mut String,
    stamps: &[i64],
    fps: u32,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    let zeros = stamps.iter().filter(|&&st| st == 0).count();
    if zeros > 0 {
        errors.push(format!("{zeros} 条消息的 stamp 为 0"));
    }
    let valid: Vec<i64> = stamps.iter().copied().filter(|&st| st != 0).collect();
    if valid.len() < 2 {
        errors.push("有效 stamp 不足 2 个，无法计算时间间隔".into());
        return;
    }

    // 按写入顺序看 stamp 是否单调递增
    let backwards = valid.windows(2).filter(|w| w[1] < w[0]).count();
    let duplicates = valid.windows(2).filter(|w| w[1] == w[0]).count();
    if backwards > 0 {
        warnings.push(format!("{backwards} 处 stamp 比前一条消息更早（乱序）"));
    }
    if duplicates > 0 {
        warnings.push(format!("{duplicates} 处 stamp 与前一条消息相同"));
    }

    let mut sorted = valid;
    sorted.sort_unstable();
    let Some(stats) = Stats::of(&intervals_ms(&sorted)) else {
        return;
    };
    let duration = (sorted[sorted.len() - 1] - sorted[0]) as f64 / 1e9;
    let hz = 1000.0 / stats.mean;
    out!(out, "  时长         : {duration:.3} s");
    out!(out, "  平均频率     : {hz:.2} Hz");
    out!(out, "  间隔 (ms)    : {stats}");
    // 留 0.5% 余量，避免 29.998 Hz 这种测量误差也报警
    if hz < fps as f64 * 0.995 {
        warnings.push(format!("平均频率 {hz:.2} Hz 低于目标 fps {fps}"));
    }
}

#[derive(Default)]
struct SyncResult {
    errors: usize,
    /// 有坏帧的 topic 数
    warnings: usize,
    episodes: usize,
    kept_frames: usize,
    reference_frames: usize,
}

/// 按 sync 配置对齐各 topic 并切分 episode，输出结果
fn check_sync(out: &mut String, reports: &[TopicReport], config: &Config, bad_frame_detail: bool) -> SyncResult {
    let sync_config = &config.sync;
    let reference_topic = sync_config.reference_topic.as_str();
    let frame_ns = 1e9 / config.fps as f64;
    out!(out, "\n同步检查");
    out!(out, "  基准         : {reference_topic}");
    out!(out, 
        "  规则         : 默认容差 {} ms，基准间隔 > {:.3} ms 处切开，少于 {} 帧的段丢弃",
        sync_config.max_dt_ms,
        sync_config.max_frame_gap * frame_ns / 1e6,
        sync_config.min_episode_frames
    );

    let stamps_of = |topic: &str| {
        let report = reports.iter().find(|r| r.topic == topic).expect("topic 来自同一份配置");
        report.stamps.clone()
    };
    let ConfigPlan { reference, topics: tolerances, missing_optional, plan } =
        match sync::plan_for_config(config, stamps_of) {
            Ok(p) => p,
            Err(e) => {
                out!(out, "  ✗ {e}");
                return SyncResult { errors: 1, ..Default::default() };
            }
        };

    out!(out, "  基准帧数     : {}", reference.len());
    for topic in &missing_optional {
        out!(out, "  - {topic}  缺失（可选），所有帧用 fill 填充，不参与对齐");
    }
    out!(out, "  |Δt|：每帧与所取消息的 stamp 之差；复用：与前一帧取到同一条消息");
    let width = tolerances.iter().map(|(t, _)| t.len()).max().unwrap_or(0);
    for ((topic, tol_ms), m) in tolerances.iter().zip(&plan.topics) {
        if *topic == reference_topic {
            continue;
        }
        if m.matched.is_empty() {
            out!(out, "  ✗ {topic:<width$}  没有有效 stamp，所有帧都是坏帧");
            continue;
        }
        let dt_ms: Vec<f64> = m.abs_dt.iter().map(|&dt| dt as f64 / 1e6).collect();
        let stats = Stats::of(&dt_ms).expect("基准帧非空");
        let reused = m.matched.windows(2).filter(|w| w[0] == w[1]).count();
        let mark = if m.bad_frames > 0 { "!" } else { "✓" };
        out!(out, 
            "  {mark} {topic:<width$}  容差 {tol_ms:>5.1}  |Δt| 平均 {:>7.3}  最大 {:>8.3} ms  复用 {reused:>4}  坏帧 {:>4}",
            stats.mean, stats.max, m.bad_frames
        );
    }

    if bad_frame_detail {
        out!(out, "  坏帧明细（帧号 @ 相对基准起点的时间：超出容差的 topic 及其 |Δt|）");
        for frame in (0..reference.len()).filter(|&f| plan.bad[f]) {
            let offenders: Vec<String> = tolerances
                .iter()
                .zip(&plan.topics)
                .filter_map(|((topic, tol_ms), m)| {
                    let dt_ms = m.abs_dt.get(frame).map_or(f64::INFINITY, |&dt| dt as f64 / 1e6);
                    (dt_ms > *tol_ms).then(|| format!("{topic} {dt_ms:.1}ms"))
                })
                .collect();
            let t = (reference[frame] - reference[0]) as f64 / 1e9;
            out!(out, "    {frame:>5} @ {t:>7.3} s: {}", offenders.join(", "));
        }
    }

    let bad_total = plan.bad.iter().filter(|&&b| b).count();
    let short_total: usize = plan.short_segments.iter().map(|r| r.len()).sum();
    out!(out, 
        "  坏帧 {bad_total}，基准断点 {}，过短丢弃 {} 段 {short_total} 帧",
        plan.gaps,
        plan.short_segments.len()
    );
    out!(out, 
        "  保留 {} / {} 帧，切分为 {} 个 episode：",
        plan.kept_frames(),
        reference.len(),
        plan.segments.len()
    );
    for (i, r) in plan.segments.iter().enumerate() {
        let duration = (reference[r.end - 1] - reference[r.start]) as f64 / 1e9;
        out!(out, "    #{i:<3} 帧 {:>5}..{:<5} {:>5} 帧  {duration:>7.3} s", r.start, r.end, r.len());
    }

    let result = SyncResult {
        errors: plan.segments.is_empty() as usize,
        warnings: plan.topics.iter().filter(|m| m.bad_frames > 0).count(),
        episodes: plan.segments.len(),
        kept_frames: plan.kept_frames(),
        reference_frames: reference.len(),
    };
    if plan.segments.is_empty() {
        out!(out, "  ✗ 没有可用的 episode");
    }
    result
}

fn join(set: &BTreeSet<String>) -> String {
    set.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
}
