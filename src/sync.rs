//! 按时间戳把各 topic 对齐到基准 topic 的帧上，并按坏帧和基准断点切分 episode

use std::ops::Range;

/// 对 `reference` 中的每个时间，找出 `stamps` 中最接近的元素下标；距离相同时取较早的一个。
///
/// 两个输入都必须已按升序排序；`stamps` 为空时返回空 Vec。
pub fn nearest(reference: &[i64], stamps: &[i64]) -> Vec<usize> {
    if stamps.is_empty() {
        return Vec::new();
    }
    let mut j = 0;
    reference
        .iter()
        .map(|&t| {
            // 前进到最后一个 <= t 的位置，reference 递增所以 j 不回退
            while j + 1 < stamps.len() && stamps[j + 1] <= t {
                j += 1;
            }
            match stamps.get(j + 1) {
                Some(&next) if (next - t) < (t - stamps[j]).abs() => j + 1,
                _ => j,
            }
        })
        .collect()
}

/// 参与对齐的一个 topic
pub struct TopicInput<'a> {
    /// 已排序的 stamp（ns）
    pub stamps: &'a [i64],
    /// 容差（ns）
    pub max_dt: i64,
}

pub struct SyncOptions {
    /// 基准相邻两帧的间隔超过该值（ns）时切开
    pub max_frame_gap: i64,
    /// 帧数少于该值的段被丢弃
    pub min_episode_frames: usize,
}

/// 单个 topic 的对齐结果，下标与基准帧一一对应
pub struct TopicMatch {
    /// 每帧取到的消息在 stamps 中的下标；topic 没有 stamp 时为空
    pub matched: Vec<usize>,
    /// 每帧的 |Δt|（ns）；topic 没有 stamp 时为空
    pub abs_dt: Vec<i64>,
    /// 因该 topic 超出容差而成为坏帧的帧数
    pub bad_frames: usize,
}

pub struct SyncPlan {
    /// 与输入 topics 一一对应
    pub topics: Vec<TopicMatch>,
    /// 任一 topic 超出容差的帧
    pub bad: Vec<bool>,
    /// 基准间隔过大导致的断点数
    pub gaps: usize,
    /// 保留的 episode，元素是基准帧的下标范围
    pub segments: Vec<Range<usize>>,
    /// 因过短被丢弃的段
    pub short_segments: Vec<Range<usize>>,
}

impl SyncPlan {
    pub fn kept_frames(&self) -> usize {
        self.segments.iter().map(|r| r.len()).sum()
    }
}

/// 以 `reference`（已排序的基准 stamp）的每一帧为准对齐各 topic，并切分 episode
pub fn plan(reference: &[i64], topics: &[TopicInput], options: &SyncOptions) -> SyncPlan {
    let mut bad = vec![false; reference.len()];
    let topics: Vec<TopicMatch> = topics
        .iter()
        .map(|input| {
            if input.stamps.is_empty() {
                bad.fill(true);
                return TopicMatch {
                    matched: Vec::new(),
                    abs_dt: Vec::new(),
                    bad_frames: reference.len(),
                };
            }
            let matched = nearest(reference, input.stamps);
            let abs_dt: Vec<i64> = matched
                .iter()
                .zip(reference)
                .map(|(&i, &t)| (input.stamps[i] - t).abs())
                .collect();
            let mut bad_frames = 0;
            for (frame, &dt) in abs_dt.iter().enumerate() {
                if dt > input.max_dt {
                    bad[frame] = true;
                    bad_frames += 1;
                }
            }
            TopicMatch { matched, abs_dt, bad_frames }
        })
        .collect();

    let mut gaps = 0;
    let mut runs: Vec<Range<usize>> = Vec::new();
    let mut start: Option<usize> = None;
    for i in 0..reference.len() {
        if bad[i] {
            if let Some(s) = start.take() {
                runs.push(s..i);
            }
            continue;
        }
        if i > 0 && reference[i] - reference[i - 1] > options.max_frame_gap {
            gaps += 1;
            if let Some(s) = start.take() {
                runs.push(s..i);
            }
        }
        start.get_or_insert(i);
    }
    if let Some(s) = start {
        runs.push(s..reference.len());
    }

    let (segments, short_segments) =
        runs.into_iter().partition(|r| r.len() >= options.min_episode_frames);
    SyncPlan { topics, bad, gaps, segments, short_segments }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_closest() {
        assert_eq!(nearest(&[0, 10, 20, 30], &[1, 12, 19, 45]), vec![0, 1, 2, 2]);
        assert_eq!(nearest(&[30], &[19, 40]), vec![1]);
    }

    #[test]
    fn tie_picks_earlier() {
        assert_eq!(nearest(&[5], &[0, 10]), vec![0]);
    }

    #[test]
    fn reference_outside_range() {
        assert_eq!(nearest(&[-100, 100], &[0, 10]), vec![0, 1]);
    }

    #[test]
    fn empty_stamps() {
        assert!(nearest(&[1, 2], &[]).is_empty());
    }

    fn options(min_episode_frames: usize) -> SyncOptions {
        SyncOptions { max_frame_gap: 15, min_episode_frames }
    }

    #[test]
    fn splits_on_bad_frames() {
        let reference = [0, 10, 20, 30, 40, 50, 60];
        // 30 附近没有消息 -> 第 3 帧是坏帧
        let stamps = [0, 10, 20, 45, 50, 60];
        let plan = plan(&reference, &[TopicInput { stamps: &stamps, max_dt: 3 }], &options(1));
        assert_eq!(plan.topics[0].bad_frames, 2); // 30 和 40 的 |Δt| 都是 5
        assert_eq!(plan.segments, vec![0..3, 5..7]);
        assert_eq!(plan.gaps, 0);
    }

    #[test]
    fn splits_on_reference_gap() {
        let reference = [0, 10, 20, 50, 60];
        let plan = plan(&reference, &[TopicInput { stamps: &reference, max_dt: 1 }], &options(1));
        assert_eq!(plan.gaps, 1);
        assert_eq!(plan.segments, vec![0..3, 3..5]);
    }

    #[test]
    fn drops_short_segments() {
        let reference = [0, 10, 20, 50, 60];
        let plan = plan(&reference, &[TopicInput { stamps: &reference, max_dt: 1 }], &options(3));
        assert_eq!(plan.segments, vec![0..3]);
        assert_eq!(plan.short_segments, vec![3..5]);
    }

    #[test]
    fn trims_edges() {
        let reference = [0, 10, 20, 30, 40];
        let stamps = [10, 20, 30];
        let plan = plan(&reference, &[TopicInput { stamps: &stamps, max_dt: 2 }], &options(1));
        assert_eq!(plan.segments, vec![1..4]);
    }
}
