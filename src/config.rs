//! 转换配置：描述 MCAP topic 如何映射到 LeRobot feature

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::{collections::HashSet, fs, path::Path};

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// 机器人型号，写入 LeRobot 的 info.json
    pub robot_type: Option<String>,

    /// 输出数据集的帧率
    pub fps: u32,

    pub sync: SyncConfig,

    /// feature 列表，顺序即输出顺序
    #[serde(rename = "feature")]
    pub features: Vec<Feature>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct SyncConfig {
    /// 时间基准 topic：它的每条消息对应数据集中的一帧，其他 topic 取 stamp 最接近的消息
    pub reference_topic: String,

    /// 默认容差（ms）：某个 topic 的 |Δt| 超过它，这一帧就是坏帧
    pub max_dt_ms: f64,

    /// 基准相邻两帧的间隔超过 max_frame_gap / fps 时，在此处切开 episode
    #[serde(default = "default_max_frame_gap")]
    pub max_frame_gap: f64,

    /// 切分后帧数少于该值的段被丢弃
    #[serde(default = "default_min_episode_frames")]
    pub min_episode_frames: usize,
}

fn default_max_frame_gap() -> f64 {
    1.5
}

fn default_min_episode_frames() -> usize {
    1
}

// 不能加 deny_unknown_fields：serde 不支持它与 flatten 同时使用，
// 未知字段由 FeatureKind 负责拒绝
#[derive(Deserialize, Debug)]
pub struct Feature {
    /// LeRobot feature 名，如 `observation.state`、`observation.images.head_left`
    pub name: String,

    #[serde(flatten)]
    pub kind: FeatureKind,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureKind {
    /// 单个图像 topic，编码成视频
    Video {
        topic: String,
        /// 覆盖 sync.max_dt_ms
        max_dt_ms: Option<f64>,
    },

    /// 由一个或多个 topic 的数值字段按顺序拼接成的 float 向量
    Vector { sources: Vec<Source> },
}

#[allow(dead_code)] // TODO: 转换逻辑实现后移除
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub topic: String,

    /// 消息内的字段路径，如 `position`、`pose.position`
    pub field: String,

    /// 每个维度的名字；为空时从消息中推断（如 JointState 的 name 字段）
    #[serde(default)]
    pub names: Vec<String>,

    /// 覆盖 sync.max_dt_ms
    pub max_dt_ms: Option<f64>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("无法读取配置 {}", path.display()))?;
        let config: Config =
            toml::from_str(&text).with_context(|| format!("配置格式错误 {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.fps > 0, "fps 必须大于 0");
        ensure!(!self.features.is_empty(), "至少需要一个 feature");

        let mut seen = HashSet::new();
        for f in &self.features {
            if !seen.insert(f.name.as_str()) {
                bail!("feature 名重复: {}", f.name);
            }
            if let FeatureKind::Vector { sources } = &f.kind {
                ensure!(!sources.is_empty(), "feature {} 的 sources 不能为空", f.name);
            }
        }
        let sync = &self.sync;
        ensure!(
            self.topics().any(|t| t == sync.reference_topic),
            "sync.reference_topic {} 没有被任何 feature 使用",
            sync.reference_topic
        );
        ensure!(sync.max_frame_gap > 1.0, "sync.max_frame_gap 必须大于 1");
        for (topic, max_dt_ms) in self.topic_tolerances() {
            ensure!(max_dt_ms > 0.0, "topic {topic} 的 max_dt_ms 必须大于 0");
        }
        Ok(())
    }

    /// 配置中引用到的所有 topic
    pub fn topics(&self) -> impl Iterator<Item = &str> {
        self.features.iter().flat_map(|f| match &f.kind {
            FeatureKind::Video { topic, .. } => vec![topic.as_str()],
            FeatureKind::Vector { sources } => sources.iter().map(|s| s.topic.as_str()).collect(),
        })
    }

    /// 每个 topic 的对齐容差（ms），按首次出现的顺序；同一 topic 多处引用时取最严格的
    pub fn topic_tolerances(&self) -> Vec<(&str, f64)> {
        let entries = self.features.iter().flat_map(|f| match &f.kind {
            FeatureKind::Video { topic, max_dt_ms } => vec![(topic.as_str(), *max_dt_ms)],
            FeatureKind::Vector { sources } => {
                sources.iter().map(|s| (s.topic.as_str(), s.max_dt_ms)).collect()
            }
        });
        let mut result: Vec<(&str, f64)> = Vec::new();
        for (topic, max_dt_ms) in entries {
            let tol = max_dt_ms.unwrap_or(self.sync.max_dt_ms);
            match result.iter_mut().find(|(t, _)| *t == topic) {
                Some((_, existing)) => *existing = existing.min(tol),
                None => result.push((topic, tol)),
            }
        }
        result
    }
}
