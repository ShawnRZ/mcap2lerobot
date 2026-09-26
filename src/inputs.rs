//! 展开输入参数：文件原样保留，文件夹递归查找其中的 .mcap 文件

use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::HashSet,
    fs,
    io::{BufRead, IsTerminal, Write},
    path::{Path, PathBuf},
};

pub struct Inputs {
    /// 去重后的文件，文件夹中找到的按路径排序
    pub files: Vec<PathBuf>,
    /// 参数中是否包含文件夹
    pub has_dir: bool,
}

pub fn collect(args: &[PathBuf]) -> Result<Inputs> {
    let mut files = Vec::new();
    let mut has_dir = false;
    for arg in args {
        let meta = fs::metadata(arg).with_context(|| format!("无法访问 {}", arg.display()))?;
        if meta.is_dir() {
            has_dir = true;
            let mut found = Vec::new();
            find_mcap(arg, &mut found)?;
            ensure!(!found.is_empty(), "文件夹 {} 中没有 .mcap 文件", arg.display());
            found.sort();
            files.extend(found);
        } else {
            files.push(arg.clone());
        }
    }
    // 同一个文件可能既被直接指定又在文件夹中，只保留第一次出现
    let mut seen = HashSet::new();
    files.retain(|f| seen.insert(fs::canonicalize(f).unwrap_or_else(|_| f.clone())));
    Ok(Inputs { files, has_dir })
}

fn find_mcap(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("无法读取文件夹 {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            find_mcap(&path, out)?;
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("mcap")) {
            out.push(path);
        }
    }
    Ok(())
}

/// 在终端中询问 [y/N]；`assume_yes` 时直接通过，非终端且未指定时报错
pub fn confirm(prompt: &str, assume_yes: bool) -> Result<bool> {
    if assume_yes {
        return Ok(true);
    }
    if !std::io::stdin().is_terminal() {
        bail!("需要确认但标准输入不是终端，请使用 --yes 跳过确认");
    }
    print!("{prompt} [y/N] ");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    Ok(matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_recursively_sorted_and_deduped() {
        let root = std::env::temp_dir().join(format!("mcap2lerobot-inputs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("sub")).unwrap();
        for f in ["b.mcap", "a.MCAP", "note.txt", "sub/c.mcap"] {
            fs::write(root.join(f), b"").unwrap();
        }
        let inputs = collect(&[root.clone(), root.join("b.mcap")]).unwrap();
        let names: Vec<_> =
            inputs.files.iter().map(|p| p.strip_prefix(&root).unwrap().to_str().unwrap().to_string()).collect();
        assert_eq!(names, vec!["a.MCAP", "b.mcap", "sub/c.mcap"]);
        assert!(inputs.has_dir);

        fs::create_dir_all(root.join("empty")).unwrap();
        assert!(collect(&[root.join("empty")]).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
