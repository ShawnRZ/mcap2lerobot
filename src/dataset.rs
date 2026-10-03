//! 写出 LeRobot 数据集
//! - v3.0：data/*.parquet、meta/episodes、meta/tasks.parquet、meta/info.json、meta/stats.json
//! - v2.1：每个 episode 一个 data parquet，meta/episodes.jsonl、meta/tasks.jsonl、meta/episodes_stats.jsonl、meta/info.json

use anyhow::{Context, Result};
use arrow::{
    array::{
        ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Float64Builder, Int64Array,
        Int64Builder, ListBuilder, StringArray, StringBuilder,
    },
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::ArrowWriter,
    basic::Compression,
    file::properties::WriterProperties,
};
use serde_json::{Map, Value as Json, json};
use std::{collections::HashMap, fs, fs::File, path::{Path, PathBuf}, sync::Arc};

use crate::stats::FeatureStats;

pub const CHUNKS_SIZE: usize = 1000;
pub const DATA_FILE_SIZE_MB: f64 = 100.0;
pub const VIDEO_FILE_SIZE_MB: f64 = 200.0;
const DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";
const VIDEO_PATH: &str = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4";
const V21_DATA_PATH: &str = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet";
const V21_VIDEO_PATH: &str = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4";

/// 输出的 LeRobot 数据集版本
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// lerobot ≥ 0.4：多个 episode 合并存放在 file-*.parquet / file-*.mp4 中
    #[value(name = "v3.0")]
    V30,
    /// lerobot 0.1 ~ 0.3：每个 episode 单独一个 parquet 和 mp4
    #[value(name = "v2.1")]
    V21,
}

impl Format {
    pub fn codebase_version(self) -> &'static str {
        match self {
            Format::V30 => "v3.0",
            Format::V21 => "v2.1",
        }
    }
}

/// v2.1 中 episode 所在的 chunk
pub fn episode_chunk(episode_index: usize) -> usize {
    episode_index / CHUNKS_SIZE
}

/// v2.1 的 episode 文件路径，如 `data/chunk-000/episode_000000.parquet`
pub fn episode_file_path(root: &Path, dir: &str, episode_index: usize, ext: &str) -> PathBuf {
    let chunk = episode_chunk(episode_index);
    root.join(dir).join(format!("chunk-{chunk:03}/episode_{episode_index:06}.{ext}"))
}

/// lerobot 自动添加的 feature，顺序固定
const DEFAULT_FEATURES: [(&str, &str); 5] = [
    ("timestamp", "float32"),
    ("frame_index", "int64"),
    ("episode_index", "int64"),
    ("index", "int64"),
    ("task_index", "int64"),
];

/// 数据集中的一个用户 feature
#[derive(Debug, Clone)]
pub enum FeatureSpec {
    Vector { name: String, names: Vec<String> },
    Video { name: String },
}

impl FeatureSpec {
    pub fn name(&self) -> &str {
        match self {
            FeatureSpec::Vector { name, .. } | FeatureSpec::Video { name } => name,
        }
    }
}

fn item_field(ty: DataType) -> Arc<Field> {
    Arc::new(Field::new("item", ty, true))
}

fn writer_props() -> WriterProperties {
    WriterProperties::builder().set_compression(Compression::SNAPPY).build()
}

fn chunk_file_path(root: &Path, dir: &str, chunk: usize, file: usize, ext: &str) -> PathBuf {
    root.join(dir).join(format!("chunk-{chunk:03}/file-{file:03}.{ext}"))
}

fn next_chunk_file(chunk: usize, file: usize) -> (usize, usize) {
    if file + 1 == CHUNKS_SIZE { (chunk + 1, 0) } else { (chunk, file + 1) }
}

/// 一个 episode 在 data parquet 中的位置；v2.1 中 file_index 即 episode_index
#[derive(Debug, Clone)]
pub struct DataLocation {
    pub chunk_index: usize,
    pub file_index: usize,
    pub from_index: usize,
    pub to_index: usize,
}

/// v3.0：写 data/chunk-*/file-*.parquet，每个 episode 一个 row group，文件超过 100MB 时换新文件；
/// v2.1：每个 episode 写一个 data/chunk-*/episode_*.parquet
pub struct DataWriter {
    root: PathBuf,
    format: Format,
    fps: u32,
    /// (feature 名, 维度)
    vectors: Vec<(String, usize)>,
    schema: SchemaRef,
    writer: Option<ArrowWriter<File>>,
    chunk_index: usize,
    file_index: usize,
    file_start_frame: usize,
    total_frames: usize,
}

impl DataWriter {
    pub fn new(root: &Path, format: Format, fps: u32, features: &[FeatureSpec]) -> Self {
        let vectors: Vec<(String, usize)> = features
            .iter()
            .filter_map(|f| match f {
                FeatureSpec::Vector { name, names } => Some((name.clone(), names.len())),
                FeatureSpec::Video { .. } => None,
            })
            .collect();
        let mut fields: Vec<Field> = vectors
            .iter()
            .map(|(name, dim)| {
                Field::new(name, DataType::FixedSizeList(item_field(DataType::Float32), *dim as i32), true)
            })
            .collect();
        for (name, dtype) in DEFAULT_FEATURES {
            let ty = if dtype == "float32" { DataType::Float32 } else { DataType::Int64 };
            fields.push(Field::new(name, ty, true));
        }
        let mut schema = Schema::new(fields);
        if format == Format::V21 {
            schema = schema.with_metadata(HashMap::from([(
                "huggingface".to_string(),
                hf_features_metadata(&vectors).to_string(),
            )]));
        }
        Self {
            root: root.to_path_buf(),
            format,
            fps,
            vectors,
            schema: Arc::new(schema),
            writer: None,
            chunk_index: 0,
            file_index: 0,
            file_start_frame: 0,
            total_frames: 0,
        }
    }

    /// `vectors` 与构造时的 vector feature 一一对应，每个是 length × dim 的按行数据
    pub fn write_episode(
        &mut self,
        episode_index: usize,
        task_index: usize,
        length: usize,
        vectors: &[Vec<f32>],
    ) -> Result<DataLocation> {
        if self.format == Format::V21 {
            if let Some(writer) = self.writer.take() {
                writer.close()?;
            }
            self.chunk_index = episode_chunk(episode_index);
            self.file_index = episode_index;
        } else if let Some(writer) = &self.writer {
            // lerobot：按当前文件的平均每帧大小估计，写入后超过限制则换新文件
            let size_mb = writer.bytes_written() as f64 / (1024.0 * 1024.0);
            let frames = self.total_frames - self.file_start_frame;
            let per_frame = if frames > 0 { size_mb / frames as f64 } else { 0.0 };
            if size_mb + per_frame * length as f64 >= DATA_FILE_SIZE_MB {
                self.writer.take().unwrap().close()?;
                (self.chunk_index, self.file_index) = next_chunk_file(self.chunk_index, self.file_index);
                self.file_start_frame = self.total_frames;
            }
        }
        if self.writer.is_none() {
            let path = match self.format {
                Format::V30 => chunk_file_path(&self.root, "data", self.chunk_index, self.file_index, "parquet"),
                Format::V21 => episode_file_path(&self.root, "data", episode_index, "parquet"),
            };
            fs::create_dir_all(path.parent().unwrap())?;
            let file = File::create(&path).with_context(|| format!("创建 {} 失败", path.display()))?;
            self.writer = Some(ArrowWriter::try_new(file, self.schema.clone(), Some(writer_props()))?);
        }

        let mut columns: Vec<ArrayRef> = Vec::new();
        for ((_, dim), values) in self.vectors.iter().zip(vectors) {
            assert_eq!(values.len(), length * dim);
            let list = FixedSizeListArray::try_new(
                item_field(DataType::Float32),
                *dim as i32,
                Arc::new(Float32Array::from(values.clone())),
                None,
            )?;
            columns.push(Arc::new(list));
        }
        let from = self.total_frames;
        columns.push(Arc::new(Float32Array::from_iter_values(
            (0..length).map(|i| i as f32 / self.fps as f32),
        )));
        columns.push(Arc::new(Int64Array::from_iter_values(0..length as i64)));
        columns.push(Arc::new(Int64Array::from(vec![episode_index as i64; length])));
        columns.push(Arc::new(Int64Array::from_iter_values(from as i64..(from + length) as i64)));
        columns.push(Arc::new(Int64Array::from(vec![task_index as i64; length])));

        let writer = self.writer.as_mut().unwrap();
        writer.write(&RecordBatch::try_new(self.schema.clone(), columns)?)?;
        // 每个 episode 一个 row group
        writer.flush()?;
        self.total_frames += length;
        Ok(DataLocation {
            chunk_index: self.chunk_index,
            file_index: self.file_index,
            from_index: from,
            to_index: from + length,
        })
    }

    pub fn finish(mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        Ok(())
    }
}

/// 一个 episode 在视频文件中的位置
#[derive(Debug, Clone)]
pub struct VideoLocation {
    pub chunk_index: usize,
    pub file_index: usize,
    pub from_timestamp: f64,
    pub to_timestamp: f64,
}

/// meta/episodes 中一行
pub struct EpisodeMeta {
    pub episode_index: usize,
    pub tasks: Vec<String>,
    pub length: usize,
    pub data: DataLocation,
    /// 与视频 feature 顺序一致
    pub videos: Vec<VideoLocation>,
    /// feature 名 -> 统计，包含用户 feature 和默认 feature
    pub stats: HashMap<String, FeatureStats>,
}

/// 默认 feature 的 episode 统计
pub fn default_feature_stats(
    fps: u32,
    episode_index: usize,
    task_index: usize,
    from_index: usize,
    length: usize,
) -> Vec<(&'static str, FeatureStats)> {
    let column = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..length).map(f).collect() };
    let n = length as u64;
    vec![
        // timestamp 以 float32 存储，统计也按 float32 取值
        ("timestamp", FeatureStats::compute(&column(&|i| (i as f32 / fps as f32) as f64), 1, n, false)),
        ("frame_index", FeatureStats::compute(&column(&|i| i as f64), 1, n, false)),
        ("episode_index", FeatureStats::compute(&column(&|_| episode_index as f64), 1, n, false)),
        ("index", FeatureStats::compute(&column(&|i| (from_index + i) as f64), 1, n, false)),
        ("task_index", FeatureStats::compute(&column(&|_| task_index as f64), 1, n, false)),
    ]
}

/// HuggingFace datasets 写 parquet 时附带的 feature 描述（schema metadata 中的 `huggingface`），
/// lerobot v2.1 用 datasets 读取 data parquet
fn hf_features_metadata(vectors: &[(String, usize)]) -> Json {
    let mut features = Map::new();
    for (name, dim) in vectors {
        features.insert(
            name.clone(),
            json!({"feature": {"dtype": "float32", "_type": "Value"}, "length": dim, "_type": "Sequence"}),
        );
    }
    for (name, dtype) in DEFAULT_FEATURES {
        features.insert(name.into(), json!({"dtype": dtype, "_type": "Value"}));
    }
    json!({"info": {"features": features}})
}

/// 写完所有 episode 后生成 meta 目录下的文件
pub struct MetaWriter<'a> {
    pub root: &'a Path,
    pub format: Format,
    pub fps: u32,
    pub robot_type: Option<&'a str>,
    pub features: &'a [FeatureSpec],
    /// 视频 feature 的 (高, 宽, info)
    pub video_info: &'a HashMap<String, (u32, u32, Map<String, Json>)>,
    pub tasks: &'a [String],
}

impl MetaWriter<'_> {
    /// 所有 feature 名，按 info.json 中的顺序
    fn all_feature_names(&self) -> Vec<&str> {
        self.features.iter().map(|f| f.name()).chain(DEFAULT_FEATURES.iter().map(|(n, _)| *n)).collect()
    }

    pub fn write(&self, episodes: &[EpisodeMeta]) -> Result<()> {
        let meta = self.root.join("meta");
        fs::create_dir_all(&meta)?;
        match self.format {
            Format::V30 => {
                self.write_tasks()?;
                self.write_episodes(episodes)?;
                write_json(&meta.join("stats.json"), &self.stats_json(episodes))?;
                write_json(&meta.join("info.json"), &self.info_json(episodes))?;
            }
            Format::V21 => {
                // v2.1 不写 stats.json，lerobot 加载时由 episodes_stats.jsonl 聚合
                self.write_v21_jsonl(episodes)?;
                write_json(&meta.join("info.json"), &self.info_json_v21(episodes))?;
            }
        }
        Ok(())
    }

    /// info.json 中的 features
    fn features_json(&self) -> Map<String, Json> {
        let mut features = Map::new();
        for f in self.features {
            let value = match f {
                FeatureSpec::Vector { names, .. } => {
                    json!({"dtype": "float32", "shape": [names.len()], "names": names})
                }
                FeatureSpec::Video { name } => {
                    let (h, w, info) = &self.video_info[name];
                    json!({
                        "dtype": "video",
                        "shape": [h, w, 3],
                        "names": ["height", "width", "channels"],
                        "info": info,
                    })
                }
            };
            features.insert(f.name().to_string(), value);
        }
        for (name, dtype) in DEFAULT_FEATURES {
            features.insert(name.into(), json!({"dtype": dtype, "shape": [1], "names": null}));
        }
        features
    }

    fn num_videos(&self) -> usize {
        self.features.iter().filter(|f| matches!(f, FeatureSpec::Video { .. })).count()
    }

    fn info_json(&self, episodes: &[EpisodeMeta]) -> Json {
        let has_video = self.num_videos() > 0;
        json!({
            "codebase_version": self.format.codebase_version(),
            "fps": self.fps,
            "features": self.features_json(),
            "total_episodes": episodes.len(),
            "total_frames": episodes.iter().map(|e| e.length).sum::<usize>(),
            "total_tasks": self.tasks.len(),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": DATA_FILE_SIZE_MB as i64,
            "video_files_size_in_mb": VIDEO_FILE_SIZE_MB as i64,
            "data_path": DATA_PATH,
            "video_path": if has_video { json!(VIDEO_PATH) } else { Json::Null },
            "robot_type": self.robot_type,
            "splits": {"train": format!("0:{}", episodes.len())},
        })
    }

    /// 与 lerobot 0.3 `create_empty_dataset_info` 的字段顺序一致
    fn info_json_v21(&self, episodes: &[EpisodeMeta]) -> Json {
        let num_videos = self.num_videos();
        json!({
            "codebase_version": self.format.codebase_version(),
            "robot_type": self.robot_type,
            "total_episodes": episodes.len(),
            "total_frames": episodes.iter().map(|e| e.length).sum::<usize>(),
            "total_tasks": self.tasks.len(),
            "total_videos": episodes.len() * num_videos,
            "total_chunks": episodes.len().div_ceil(CHUNKS_SIZE),
            "chunks_size": CHUNKS_SIZE,
            "fps": self.fps,
            "splits": {"train": format!("0:{}", episodes.len())},
            "data_path": V21_DATA_PATH,
            "video_path": if num_videos > 0 { json!(V21_VIDEO_PATH) } else { Json::Null },
            "features": self.features_json(),
        })
    }

    /// meta/tasks.jsonl、meta/episodes.jsonl、meta/episodes_stats.jsonl
    fn write_v21_jsonl(&self, episodes: &[EpisodeMeta]) -> Result<()> {
        let meta = self.root.join("meta");
        let tasks = self.tasks.iter().enumerate().map(|(i, t)| json!({"task_index": i, "task": t}));
        write_jsonl(&meta.join("tasks.jsonl"), tasks)?;
        let rows = episodes
            .iter()
            .map(|e| json!({"episode_index": e.episode_index, "tasks": e.tasks, "length": e.length}));
        write_jsonl(&meta.join("episodes.jsonl"), rows)?;
        let stats = episodes.iter().map(|e| {
            // v2.1 的统计只有 min/max/mean/std/count
            let stats: Map<String, Json> =
                self.all_feature_names().into_iter().map(|n| (n.to_string(), e.stats[n].to_json(false))).collect();
            json!({"episode_index": e.episode_index, "stats": stats})
        });
        write_jsonl(&meta.join("episodes_stats.jsonl"), stats)
    }

    fn stats_json(&self, episodes: &[EpisodeMeta]) -> Json {
        let mut out = Map::new();
        for name in self.all_feature_names() {
            let list: Vec<&FeatureStats> = episodes.iter().map(|e| &e.stats[name]).collect();
            if !list.is_empty() {
                out.insert(name.into(), FeatureStats::aggregate(&list).to_json(true));
            }
        }
        Json::Object(out)
    }

    /// pandas DataFrame（index 为 task，列为 task_index）经 pyarrow 写出的格式
    fn write_tasks(&self) -> Result<()> {
        let pandas = json!({
            "index_columns": ["task"],
            "column_indexes": [{
                "name": null, "field_name": null, "pandas_type": "unicode",
                "numpy_type": "object", "metadata": {"encoding": "UTF-8"}
            }],
            "columns": [
                {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                 "numpy_type": "int64", "metadata": null},
                {"name": "task", "field_name": "task", "pandas_type": "unicode",
                 "numpy_type": "object", "metadata": null},
            ],
            "creator": {"library": "pyarrow", "version": "21.0.0"},
            "pandas_version": "2.3.3",
        });
        let schema = Schema::new(vec![
            Field::new("task_index", DataType::Int64, true),
            Field::new("task", DataType::Utf8, true),
        ])
        .with_metadata(HashMap::from([("pandas".to_string(), pandas.to_string())]));
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..self.tasks.len() as i64)),
                Arc::new(StringArray::from_iter_values(self.tasks)),
            ],
        )?;
        write_parquet(&self.root.join("meta/tasks.parquet"), &batch)
    }

    fn write_episodes(&self, episodes: &[EpisodeMeta]) -> Result<()> {
        let mut cols = Columns::default();
        let int = |f: &dyn Fn(&EpisodeMeta) -> usize| -> ArrayRef {
            Arc::new(Int64Array::from_iter_values(episodes.iter().map(|e| f(e) as i64)))
        };
        let float = |f: &dyn Fn(&EpisodeMeta) -> f64| -> ArrayRef {
            Arc::new(Float64Array::from_iter_values(episodes.iter().map(f)))
        };

        cols.push("episode_index", int(&|e| e.episode_index));
        let mut tasks = ListBuilder::new(StringBuilder::new());
        for e in episodes {
            for t in &e.tasks {
                tasks.values().append_value(t);
            }
            tasks.append(true);
        }
        cols.push("tasks", Arc::new(tasks.finish()));
        cols.push("length", int(&|e| e.length));
        cols.push("data/chunk_index", int(&|e| e.data.chunk_index));
        cols.push("data/file_index", int(&|e| e.data.file_index));
        cols.push("dataset_from_index", int(&|e| e.data.from_index));
        cols.push("dataset_to_index", int(&|e| e.data.to_index));

        let video_names = self.features.iter().filter(|f| matches!(f, FeatureSpec::Video { .. }));
        for (vi, f) in video_names.enumerate() {
            let key = f.name();
            cols.push(&format!("videos/{key}/chunk_index"), int(&|e| e.videos[vi].chunk_index));
            cols.push(&format!("videos/{key}/file_index"), int(&|e| e.videos[vi].file_index));
            cols.push(&format!("videos/{key}/from_timestamp"), float(&|e| e.videos[vi].from_timestamp));
            cols.push(&format!("videos/{key}/to_timestamp"), float(&|e| e.videos[vi].to_timestamp));
        }

        for name in self.all_feature_names() {
            let stats: Vec<&FeatureStats> = episodes.iter().map(|e| &e.stats[name]).collect();
            let image = stats.first().is_some_and(|s| s.image);
            let stat_names = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"];
            for stat in stat_names {
                let col = format!("stats/{name}/{stat}");
                if stat == "count" {
                    let mut b = ListBuilder::new(Int64Builder::new());
                    for s in &stats {
                        b.values().append_value(s.count as i64);
                        b.append(true);
                    }
                    cols.push(&col, Arc::new(b.finish()));
                    continue;
                }
                let values = |s: &FeatureStats| -> Vec<f64> {
                    s.named_values().into_iter().find(|(k, _)| *k == stat).unwrap().1.clone()
                };
                if image {
                    // (C, 1, 1)
                    let mut b = ListBuilder::new(ListBuilder::new(ListBuilder::new(Float64Builder::new())));
                    for s in &stats {
                        for v in values(s) {
                            b.values().values().values().append_value(v);
                            b.values().values().append(true);
                            b.values().append(true);
                        }
                        b.append(true);
                    }
                    cols.push(&col, Arc::new(b.finish()));
                } else {
                    let mut b = ListBuilder::new(Float64Builder::new());
                    for s in &stats {
                        b.values().append_slice(&values(s));
                        b.append(true);
                    }
                    cols.push(&col, Arc::new(b.finish()));
                }
            }
        }

        cols.push("meta/episodes/chunk_index", int(&|_| 0));
        cols.push("meta/episodes/file_index", int(&|_| 0));
        let batch = cols.into_batch()?;
        write_parquet(&chunk_file_path(self.root, "meta/episodes", 0, 0, "parquet"), &batch)
    }
}

#[derive(Default)]
struct Columns {
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
}

impl Columns {
    fn push(&mut self, name: &str, array: ArrayRef) {
        self.fields.push(Field::new(name, array.data_type().clone(), true));
        self.arrays.push(array);
    }

    fn into_batch(self) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(Arc::new(Schema::new(self.fields)), self.arrays)?)
    }
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let file = File::create(path).with_context(|| format!("创建 {} 失败", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(writer_props()))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

fn write_jsonl(path: &Path, rows: impl Iterator<Item = Json>) -> Result<()> {
    let mut buf = String::new();
    for row in rows {
        buf.push_str(&serde_json::to_string(&row)?);
        buf.push('\n');
    }
    fs::write(path, buf).with_context(|| format!("写入 {} 失败", path.display()))
}

fn write_json(path: &Path, value: &Json) -> Result<()> {
    // 与 lerobot 的 json.dump(indent=4, ensure_ascii=False) 一致
    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    serde::Serialize::serialize(value, &mut ser)?;
    fs::write(path, buf).with_context(|| format!("写入 {} 失败", path.display()))
}
