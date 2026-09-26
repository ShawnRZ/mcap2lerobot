use anyhow::{Result, bail};
use clap::Args;
use mcap::Summary;
use std::{collections::HashSet, path::PathBuf};

use crate::config::Config;
use crate::map_file;

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// 输入的 MCAP 文件
    input: PathBuf,

    /// topic -> feature 映射配置文件（TOML）
    #[arg(short, long)]
    config: PathBuf,

    /// 输出目录
    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    /// 目标帧率，覆盖配置文件中的 fps
    #[arg(long)]
    fps: Option<u32>,

    /// 输出目录已存在时覆盖
    #[arg(long)]
    force: bool,
}

pub fn run(args: &ConvertArgs, _verbose: u8) -> Result<()> {
    let mut config = Config::load(&args.config)?;
    if let Some(fps) = args.fps {
        config.fps = fps;
    }

    if args.output.exists() && !args.force {
        bail!("输出目录 {} 已存在，使用 --force 覆盖", args.output.display());
    }

    let mapped = map_file(&args.input)?;

    // 检查配置里的 topic 是否都存在于 MCAP 中
    if let Some(summary) = Summary::read(&mapped)? {
        let available: HashSet<&str> = summary.channels.values().map(|c| c.topic.as_str()).collect();
        let missing: Vec<&str> = config.topics().filter(|t| !available.contains(t)).collect();
        if !missing.is_empty() {
            bail!("MCAP 中找不到以下 topic: {missing:?}");
        }
    }

    // TODO: 实现转换逻辑
    println!(
        "convert {} -> {} (robot: {:?}, fps: {})",
        args.input.display(),
        args.output.display(),
        config.robot_type,
        config.fps
    );
    for f in &config.features {
        println!("  {}: {:?}", f.name, f.kind);
    }

    Ok(())
}
