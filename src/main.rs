mod cdr;
mod config;
mod convert;
mod info;
mod sync;
mod verify;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use memmap2::Mmap;
use std::{fs, path::Path};

/// 将 MCAP 文件转换为 LeRobot 数据集
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// 输出更详细的信息（可重复：-v, -vv）
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// 查看 MCAP 文件的概要信息（消息数、时间范围、channel 列表）
    Info(info::InfoArgs),
    /// 检查配置中引用的 topic：schema、图像编码、消息时间间隔
    Verify(verify::VerifyArgs),
    /// 将 MCAP 文件转换为 LeRobot 格式
    Convert(convert::ConvertArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Info(args) => info::run(&args, cli.verbose),
        Command::Verify(args) => verify::run(&args, cli.verbose),
        Command::Convert(args) => convert::run(&args, cli.verbose),
    }
}

fn map_file(path: &Path) -> Result<Mmap> {
    let fd = fs::File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    // SAFETY: 转换期间不应有其他进程修改该文件
    let mapped = unsafe { Mmap::map(&fd) }.with_context(|| format!("无法映射 {}", path.display()))?;
    Ok(mapped)
}
