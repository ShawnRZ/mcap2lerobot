use anyhow::Result;
use clap::Args;
use mcap::{MessageStream, Summary};
use std::path::PathBuf;

use crate::map_file;

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// 输入的 MCAP 文件
    input: PathBuf,

    /// 逐条打印消息
    #[arg(long)]
    messages: bool,
}

pub fn run(args: &InfoArgs, _verbose: u8) -> Result<()> {
    let mapped = map_file(&args.input)?;

    if let Some(summary) = Summary::read(&mapped)? {
        if let Some(stats) = &summary.stats {
            println!(
                "消息总数 {}, 时间范围 {}..{}",
                stats.message_count, stats.message_start_time, stats.message_end_time
            );
        }
        for (id, ch) in &summary.channels {
            let count = summary
                .stats
                .as_ref()
                .and_then(|s| s.channel_message_counts.get(id));
            println!(
                "channel {id}: {} ({:?}) messages={}",
                ch.topic,
                ch.schema.as_ref().map(|s| &s.name),
                count.map_or("?".to_string(), |c| c.to_string())
            );
        }
    } else {
        println!("文件没有 summary 段");
    }

    if args.messages {
        for msg in MessageStream::new(&mapped)? {
            let msg = msg?;
            let ch = &msg.channel;
            println!(
                "topic={} encoding={} schema={:?} log_time={} bytes={}",
                ch.topic,
                ch.message_encoding,
                ch.schema.as_ref().map(|s| &s.name),
                msg.log_time,
                msg.data.len()
            );
        }
    }

    Ok(())
}
