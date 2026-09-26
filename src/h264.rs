//! H.264 Annex-B 码流的最小解析：NAL 类型、关键帧和 B 帧判断

const NAL_SLICE: u8 = 1;
const NAL_IDR: u8 = 5;

/// 按起始码（00 00 01 / 00 00 00 01）切分出各 NAL 单元（不含起始码）
pub fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let end = starts.get(k + 1).map_or(data.len(), |&next| next - 3);
            // 4 字节起始码的前导 0 属于上一个 NAL 的末尾，去掉
            let mut nal = &data[s..end];
            while nal.last() == Some(&0) && k + 1 < starts.len() {
                nal = &nal[..nal.len() - 1];
            }
            nal
        })
        .filter(|nal| !nal.is_empty())
        .collect()
}

fn nal_type(nal: &[u8]) -> u8 {
    nal[0] & 0x1f
}

/// 包含 IDR 图像
pub fn is_keyframe(data: &[u8]) -> bool {
    nal_units(data).iter().any(|n| nal_type(n) == NAL_IDR)
}

/// 包含 B slice
pub fn has_b_slice(data: &[u8]) -> bool {
    nal_units(data)
        .iter()
        .filter(|n| matches!(nal_type(n), NAL_SLICE | NAL_IDR))
        .any(|n| slice_type(n).is_some_and(|t| t % 5 == 1))
}

/// slice header 开头：first_mb_in_slice ue(v)、slice_type ue(v)
fn slice_type(nal: &[u8]) -> Option<u32> {
    // 去掉防竞争字节 00 00 03，只需要开头几个字节
    let mut rbsp = Vec::with_capacity(16);
    for &b in nal[1..].iter().take(16) {
        if b == 3 && rbsp.ends_with(&[0, 0]) {
            continue;
        }
        rbsp.push(b);
    }
    let mut bits = BitReader { data: &rbsp, pos: 0 };
    bits.ue()?;
    bits.ue()
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.pos / 8)?;
        let bit = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Some(bit as u32)
    }

    /// 无符号指数哥伦布编码
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let mut value = 0;
        for _ in 0..zeros {
            value = (value << 1) | self.bit()?;
        }
        Some((1 << zeros) - 1 + value)
    }
}

/// 码流从第一个关键帧开始才能解码；之前的消息应丢弃
#[derive(Default)]
pub struct KeyframeGate {
    started: bool,
}

impl KeyframeGate {
    /// 该消息能否送入解码器
    pub fn accept(&mut self, data: &[u8]) -> bool {
        if !self.started && is_keyframe(data) {
            self.started = true;
        }
        self.started
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // slice header 位串：first_mb=0 -> "1"，slice_type=n -> ue(n)
    // P(0)="1", B(1)="010", I(2)="011", P(5)="00110", B(6)="00111", I(7)="0001000"
    const P_SLICE: &[u8] = &[0, 0, 0, 1, 0x41, 0b1_1_000000];
    const B_SLICE: &[u8] = &[0, 0, 1, 0x01, 0b1_00111_00];
    const IDR: &[u8] = &[0, 0, 0, 1, 0x67, 0x64, 0, 0, 0, 1, 0x68, 0xee, 0, 0, 1, 0x65, 0b1_0001000];

    #[test]
    fn splits_nal_units() {
        let types: Vec<u8> = nal_units(IDR).iter().map(|n| nal_type(n)).collect();
        assert_eq!(types, vec![7, 8, 5]);
        assert_eq!(nal_units(IDR)[0], &[0x67, 0x64][..]);
    }

    #[test]
    fn detects_keyframe_and_b_slice() {
        assert!(is_keyframe(IDR));
        assert!(!is_keyframe(P_SLICE));
        assert!(!has_b_slice(P_SLICE));
        assert!(!has_b_slice(IDR));
        assert!(has_b_slice(B_SLICE));
    }

    #[test]
    fn gate_waits_for_keyframe() {
        let mut gate = KeyframeGate::default();
        assert!(!gate.accept(P_SLICE));
        assert!(gate.accept(IDR));
        assert!(gate.accept(P_SLICE));
    }
}
