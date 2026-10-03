//! 与 lerobot `compute_stats.py` 一致的 feature 统计：min/max/mean/std/count 和直方图估计的分位数

use serde_json::{Value as Json, json};

pub const QUANTILES: [f64; 5] = [0.01, 0.10, 0.50, 0.90, 0.99];
pub const QUANTILE_KEYS: [&str; 5] = ["q01", "q10", "q50", "q90", "q99"];
const NUM_QUANTILE_BINS: usize = 5000;

/// 单个 feature 的统计，每个统计量的长度等于维度数（图像为通道数）
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub count: u64,
    pub quantiles: [Vec<f64>; 5],
    /// 图像统计，序列化为 (C, 1, 1)
    pub image: bool,
}

impl FeatureStats {
    /// `rows` 为 n 行 dim 列、按行存放的数据；`count` 为写入统计的样本数
    pub fn compute(rows: &[f64], dim: usize, count: u64, image: bool) -> Self {
        let n = rows.len() / dim;
        assert!(n > 0 && rows.len() == n * dim, "统计数据形状错误");
        let column = |d: usize| rows.iter().skip(d).step_by(dim).copied();

        let min: Vec<f64> = (0..dim).map(|d| column(d).fold(f64::INFINITY, f64::min)).collect();
        let max: Vec<f64> = (0..dim).map(|d| column(d).fold(f64::NEG_INFINITY, f64::max)).collect();
        let mean: Vec<f64> = (0..dim).map(|d| column(d).sum::<f64>() / n as f64).collect();

        if n < 2 {
            // lerobot 在样本不足时用 np.std，分位数取均值
            let std = (0..dim)
                .map(|d| (column(d).map(|x| (x - mean[d]).powi(2)).sum::<f64>() / n as f64).sqrt())
                .collect();
            let quantiles = std::array::from_fn(|_| mean.clone());
            return Self { min, max, mean, std, count, quantiles, image };
        }

        // RunningQuantileStats：std = sqrt(max(0, E[x²] - E[x]²))
        let std = (0..dim)
            .map(|d| {
                let mean_sq = column(d).map(|x| x * x).sum::<f64>() / n as f64;
                (mean_sq - mean[d] * mean[d]).max(0.0).sqrt()
            })
            .collect();

        let mut quantiles: [Vec<f64>; 5] = Default::default();
        for d in 0..dim {
            let lo = min[d] - 1e-10;
            let hi = max[d] + 1e-10;
            let width = (hi - lo) / NUM_QUANTILE_BINS as f64;
            let mut hist = vec![0u64; NUM_QUANTILE_BINS];
            for x in column(d) {
                let bin = ((x - lo) / width) as usize;
                hist[bin.min(NUM_QUANTILE_BINS - 1)] += 1;
            }
            let mut cumsum = hist;
            for i in 1..cumsum.len() {
                cumsum[i] += cumsum[i - 1];
            }
            for (qi, q) in QUANTILES.iter().enumerate() {
                quantiles[qi].push(histogram_quantile(&cumsum, lo, width, q * n as f64));
            }
        }
        Self { min, max, mean, std, count, quantiles, image }
    }

    /// 由每个通道 0..=255 的像素计数计算统计，等价于把所有像素逐个送入 [`FeatureStats::compute`]；
    /// `count` 为抽样的图像数（lerobot 的图像 count 是图像数而不是像素数）
    pub fn from_pixel_counts(counts: &[[u64; 256]], count: u64) -> Self {
        let dim = counts.len();
        let mut stats = Self {
            min: Vec::with_capacity(dim),
            max: Vec::with_capacity(dim),
            mean: Vec::with_capacity(dim),
            std: Vec::with_capacity(dim),
            count,
            quantiles: Default::default(),
            image: true,
        };
        for hist in counts {
            let values = || (0..256usize).filter(|&v| hist[v] > 0);
            let n: u64 = hist.iter().sum();
            let min = values().next().expect("至少有一个像素") as f64;
            let max = values().last().unwrap() as f64;
            let mean = values().map(|v| v as f64 * hist[v] as f64).sum::<f64>() / n as f64;
            let mean_sq = values().map(|v| (v * v) as f64 * hist[v] as f64).sum::<f64>() / n as f64;

            let lo = min - 1e-10;
            let hi = max + 1e-10;
            let width = (hi - lo) / NUM_QUANTILE_BINS as f64;
            let mut cumsum = vec![0u64; NUM_QUANTILE_BINS];
            for v in values() {
                let bin = ((v as f64 - lo) / width) as usize;
                cumsum[bin.min(NUM_QUANTILE_BINS - 1)] += hist[v];
            }
            for i in 1..cumsum.len() {
                cumsum[i] += cumsum[i - 1];
            }
            for (qi, q) in QUANTILES.iter().enumerate() {
                stats.quantiles[qi].push(histogram_quantile(&cumsum, lo, width, q * n as f64));
            }
            stats.min.push(min);
            stats.max.push(max);
            stats.mean.push(mean);
            stats.std.push((mean_sq - mean * mean).max(0.0).sqrt());
        }
        stats
    }

    /// 所有统计量（count 除外）除以 factor，用于把图像像素值归一化到 [0, 1]
    pub fn scale(mut self, factor: f64) -> Self {
        for v in [&mut self.min, &mut self.max, &mut self.mean, &mut self.std]
            .into_iter()
            .chain(self.quantiles.iter_mut())
        {
            v.iter_mut().for_each(|x| *x /= factor);
        }
        self
    }

    /// 按 count 加权合并多个 episode 的统计（lerobot `aggregate_feature_stats`）
    pub fn aggregate(list: &[&FeatureStats]) -> Self {
        let first = list[0];
        let dim = first.mean.len();
        let total: f64 = list.iter().map(|s| s.count as f64).sum();
        let weighted = |f: &dyn Fn(&FeatureStats) -> &Vec<f64>| -> Vec<f64> {
            (0..dim).map(|d| list.iter().map(|s| f(s)[d] * s.count as f64).sum::<f64>() / total).collect()
        };
        let mean = weighted(&|s| &s.mean);
        let std = (0..dim)
            .map(|d| {
                let var: f64 = list
                    .iter()
                    .map(|s| (s.std[d].powi(2) + (s.mean[d] - mean[d]).powi(2)) * s.count as f64)
                    .sum();
                (var / total).sqrt()
            })
            .collect();
        let fold = |f: &dyn Fn(&FeatureStats) -> &Vec<f64>, pick: fn(f64, f64) -> f64| -> Vec<f64> {
            (0..dim).map(|d| list.iter().map(|s| f(s)[d]).reduce(pick).unwrap()).collect()
        };
        let quantiles = std::array::from_fn(|qi| weighted(&|s| &s.quantiles[qi]));
        Self {
            min: fold(&|s| &s.min, f64::min),
            max: fold(&|s| &s.max, f64::max),
            mean,
            std,
            count: total as u64,
            quantiles,
            image: first.image,
        }
    }

    /// (统计量名, 值) 按 lerobot 的顺序：min, max, mean, std, count, q01..q99；count 单独处理
    pub fn named_values(&self) -> Vec<(&'static str, &Vec<f64>)> {
        let mut out = vec![("min", &self.min), ("max", &self.max), ("mean", &self.mean), ("std", &self.std)];
        out.extend(QUANTILE_KEYS.iter().zip(&self.quantiles).map(|(k, v)| (*k, v)));
        out
    }

    /// 按形状包装：图像为 [[[v]], ...]，其他为 [v, ...]
    pub fn shape_json(&self, values: &[f64]) -> Json {
        if self.image { json!(values.iter().map(|v| [[*v]]).collect::<Vec<_>>()) } else { json!(values) }
    }

    /// `quantiles` 为 false 时只输出 min/max/mean/std/count（LeRobot v2.1）
    pub fn to_json(&self, quantiles: bool) -> Json {
        let mut map = serde_json::Map::new();
        for (k, v) in self.named_values().into_iter().take(4) {
            map.insert(k.into(), self.shape_json(v));
        }
        map.insert("count".into(), json!([self.count]));
        if !quantiles {
            return Json::Object(map);
        }
        for (k, v) in QUANTILE_KEYS.iter().zip(&self.quantiles) {
            map.insert((*k).into(), self.shape_json(v));
        }
        Json::Object(map)
    }
}

/// lerobot `RunningQuantileStats._compute_single_quantile`：在累计直方图上线性插值
fn histogram_quantile(cumsum: &[u64], lo: f64, width: f64, target: f64) -> f64 {
    let edge = |i: usize| lo + width * i as f64;
    // np.searchsorted(cumsum, target)：第一个 >= target 的位置
    let idx = cumsum.partition_point(|&c| (c as f64) < target);
    if idx == 0 {
        return edge(0);
    }
    if idx >= cumsum.len() {
        return edge(cumsum.len());
    }
    let before = cumsum[idx - 1] as f64;
    let in_bin = cumsum[idx] as f64 - before;
    if in_bin == 0.0 { edge(idx) } else { edge(idx) + (target - before) / in_bin * width }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_stats() {
        let rows: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let s = FeatureStats::compute(&rows, 1, 101, false);
        assert_eq!(s.min, vec![0.0]);
        assert_eq!(s.max, vec![100.0]);
        assert_eq!(s.mean, vec![50.0]);
        assert!((s.std[0] - 29.1547594742265).abs() < 1e-9);
        assert!((s.quantiles[2][0] - 50.0).abs() < 0.05);
        assert!((s.quantiles[0][0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn aggregate_matches_combined() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (50..100).map(|i| i as f64).collect();
        let all: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let sa = FeatureStats::compute(&a, 1, 50, false);
        let sb = FeatureStats::compute(&b, 1, 50, false);
        let agg = FeatureStats::aggregate(&[&sa, &sb]);
        let direct = FeatureStats::compute(&all, 1, 100, false);
        assert!((agg.mean[0] - direct.mean[0]).abs() < 1e-9);
        assert!((agg.std[0] - direct.std[0]).abs() < 1e-9);
        assert_eq!(agg.count, 100);
        assert_eq!((agg.min[0], agg.max[0]), (0.0, 99.0));
    }

    #[test]
    fn pixel_counts_match_compute() {
        let pixels: Vec<f64> = (0..1000).map(|i| ((i * 37) % 256) as f64).collect();
        let mut counts = [[0u64; 256]; 1];
        for p in &pixels {
            counts[0][*p as usize] += 1;
        }
        let a = FeatureStats::from_pixel_counts(&counts, 7);
        let b = FeatureStats::compute(&pixels, 1, 7, true);
        for (x, y) in a.named_values().iter().zip(b.named_values()) {
            assert!((x.1[0] - y.1[0]).abs() < 1e-9, "{} {} {}", x.0, x.1[0], y.1[0]);
        }
    }

    #[test]
    fn single_sample() {
        let s = FeatureStats::compute(&[3.0, 4.0], 2, 1, false);
        assert_eq!(s.std, vec![0.0, 0.0]);
        assert_eq!(s.quantiles[4], vec![3.0, 4.0]);
    }
}
