use anyhow::{Result, bail};
use clap::Args;
use mcap::{MessageStream, Schema};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::PathBuf,
};

use crate::cdr::CdrReader;
use crate::config::{Config, FeatureKind};
use crate::map_file;
use crate::sync::nearest;

const COMPRESSED_IMAGE: &str = "sensor_msgs/msg/CompressedImage";
const RAW_IMAGE: &str = "sensor_msgs/msg/Image";

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// 输入的 MCAP 文件
    input: PathBuf,

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

pub fn run(args: &VerifyArgs, _verbose: u8) -> Result<()> {
    let config = Config::load(&args.config)?;
    let mut reports = collect_topics(&config);
    let index: HashMap<&str, usize> =
        reports.iter().enumerate().map(|(i, r)| (r.topic, i)).collect();
    // channel id -> schema 是否带 Header
    let mut header_cache: HashMap<u16, bool> = HashMap::new();

    let mapped = map_file(&args.input)?;
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
            .or_insert_with(|| schema.is_some_and(schema_has_header));

        report.schemas.insert(schema_name.to_string());
        report.message_encodings.insert(msg.channel.message_encoding.clone());
        report.has_header |= has_header;
        report.message_count += 1;

        if has_header && msg.channel.message_encoding == "cdr" {
            match decode(schema_name, report.kind, &msg.data) {
                Ok((stamp, encoding)) => {
                    report.stamps.push(stamp);
                    if let Some(enc) = encoding {
                        *report.image_encodings.entry(enc).or_default() += 1;
                    }
                }
                Err(e) => {
                    report.decode_errors += 1;
                    report.first_decode_error.get_or_insert_with(|| format!("{e:#}"));
                }
            }
        }
    }

    let mut error_count = 0;
    for report in &reports {
        error_count += print_report(report, config.fps);
    }
    error_count += check_alignment(&reports, &config.reference_topic);

    println!();
    if error_count > 0 {
        bail!("verify 失败：共 {error_count} 个问题");
    }
    println!("verify 通过：{} 个 topic 全部正常", reports.len());
    Ok(())
}

/// 按配置中首次出现的顺序收集 topic；同一 topic 被多个 feature 引用时，只要有一个是 video 就按 video 检查
fn collect_topics(config: &Config) -> Vec<TopicReport<'_>> {
    let mut reports: Vec<TopicReport> = Vec::new();
    for feature in &config.features {
        let entries: Vec<(&str, TopicKind)> = match &feature.kind {
            FeatureKind::Video { topic } => vec![(topic, TopicKind::Video)],
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
    reports
}

/// 根据 ros2msg schema 文本判断第一个字段是否为 std_msgs/Header
fn schema_has_header(schema: &Schema) -> bool {
    if schema.encoding != "ros2msg" {
        return false;
    }
    let Ok(text) = std::str::from_utf8(&schema.data) else {
        return false;
    };
    text.lines()
        // 依赖的子消息定义从 "====" 分隔线开始
        .take_while(|line| !line.starts_with("===="))
        .map(|line| line.split('#').next().unwrap_or("").trim())
        // 常量（如 `uint8 FOO=1`）不参与序列化
        .find(|line| !line.is_empty() && !line.contains('='))
        .and_then(|line| line.split_whitespace().next())
        .is_some_and(|ty| ty == "std_msgs/Header" || ty == "Header")
}

/// 解析带 Header 的 CDR 消息，返回 stamp（ns）；video topic 同时返回图像编码
fn decode(schema: &str, kind: TopicKind, data: &[u8]) -> Result<(i64, Option<String>)> {
    let mut r = CdrReader::new(data)?;
    let stamp = r.read_header()?;
    if kind != TopicKind::Video {
        return Ok((stamp, None));
    }
    let encoding = match schema {
        COMPRESSED_IMAGE => Some(r.read_string()?.to_string()),
        RAW_IMAGE => {
            let height = r.read_u32()?;
            let width = r.read_u32()?;
            let encoding = r.read_string()?;
            Some(format!("{encoding} {width}x{height}"))
        }
        // 非图像 schema，由 print_report 统一报错
        _ => None,
    };
    Ok((stamp, encoding))
}

/// 打印单个 topic 的检查结果，返回问题数
fn print_report(report: &TopicReport, fps: u32) -> usize {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let kind = match report.kind {
        TopicKind::Video => "video",
        TopicKind::Vector => "vector",
    };
    println!("\n[{kind}] {}", report.topic);
    println!("  feature      : {}", report.features.join(", "));

    if report.message_count == 0 {
        println!("  ✗ 没有任何消息（topic 不存在或为空）");
        return 1;
    }

    println!("  schema       : {}", join(&report.schemas));
    println!("  消息编码     : {}", join(&report.message_encodings));
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
        println!("  图像编码     : {encodings}");
        if report.image_encodings.len() > 1 {
            errors.push("图像编码不一致".into());
        }
    }
    if report.decode_errors > 0 {
        errors.push(format!(
            "{} 条消息解析失败，首个错误: {}",
            report.decode_errors,
            report.first_decode_error.as_deref().unwrap_or("")
        ));
    }

    println!("  消息数       : {}", report.message_count);
    if report.has_header {
        check_stamps(&report.stamps, fps, &mut errors, &mut warnings);
    } else {
        errors.push("schema 没有 Header，无法获取 header.stamp".into());
    }

    for w in &warnings {
        println!("  ! {w}");
    }
    for e in &errors {
        println!("  ✗ {e}");
    }
    if warnings.is_empty() && errors.is_empty() {
        println!("  ✓ OK");
    }
    errors.len()
}

/// 检查 header.stamp：0 值、单调性、间隔和频率；stamps 按消息在文件中的顺序排列
fn check_stamps(stamps: &[i64], fps: u32, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
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
    println!("  时长         : {duration:.3} s");
    println!("  平均频率     : {hz:.2} Hz");
    println!("  间隔 (ms)    : {stats}");
    if hz < fps as f64 {
        warnings.push(format!("平均频率 {hz:.2} Hz 低于目标 fps {fps}"));
    }
}

/// 以基准 topic 的每一帧为准，统计其他 topic 最近消息的时间偏差，返回问题数
fn check_alignment(reports: &[TopicReport], reference_topic: &str) -> usize {
    let valid_sorted = |r: &TopicReport| {
        let mut v: Vec<i64> = r.stamps.iter().copied().filter(|&st| st != 0).collect();
        v.sort_unstable();
        v
    };

    println!("\n对齐检查（基准 {reference_topic}）");
    let reference = reports
        .iter()
        .find(|r| r.topic == reference_topic)
        .map(valid_sorted)
        .unwrap_or_default();
    if reference.len() < 2 {
        println!("  ✗ 基准 topic 的有效 stamp 不足 2 个");
        return 1;
    }
    let frame_ms =
        (reference[reference.len() - 1] - reference[0]) as f64 / 1e6 / (reference.len() - 1) as f64;
    println!("  帧数 {}，平均帧间隔 {frame_ms:.3} ms", reference.len());
    println!("  |Δt|：每帧与所取消息的 stamp 之差；复用：与前一帧取到同一条消息");

    let others: Vec<&TopicReport> = reports.iter().filter(|r| r.topic != reference_topic).collect();
    let width = others.iter().map(|r| r.topic.len()).max().unwrap_or(0);
    let mut errors = 0;
    for report in others {
        let topic = report.topic;
        let stamps = valid_sorted(report);
        if stamps.is_empty() {
            println!("  {topic:<width$}  ✗ 没有有效 stamp");
            errors += 1;
            continue;
        }
        let matched = nearest(&reference, &stamps);
        let offsets: Vec<f64> = matched
            .iter()
            .zip(&reference)
            .map(|(&i, &t)| (stamps[i] - t).abs() as f64 / 1e6)
            .collect();
        let stats = Stats::of(&offsets).expect("reference 非空");
        let reused = matched.windows(2).filter(|w| w[0] == w[1]).count();
        let over_half = offsets.iter().filter(|&&x| x > frame_ms / 2.0).count();
        let mark = if over_half > 0 { "!" } else { "✓" };
        println!(
            "  {mark} {topic:<width$}  |Δt| 平均 {:>7.3}  最大 {:>8.3} ms  复用 {reused:>4} 帧  超过半帧 {over_half:>4} 帧",
            stats.mean, stats.max
        );
    }
    errors
}

fn join(set: &BTreeSet<String>) -> String {
    set.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
}
