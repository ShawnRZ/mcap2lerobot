//! 按时间戳把各 topic 对齐到基准 topic 的帧上

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

#[cfg(test)]
mod tests {
    use super::nearest;

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
}
