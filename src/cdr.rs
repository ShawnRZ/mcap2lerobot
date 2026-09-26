//! ROS2 CDR（XCDR1）读取器：按顺序读取基本类型、字符串和字节序列

use anyhow::{Context, Result, bail, ensure};

pub struct CdrReader<'a> {
    /// 去掉 4 字节封装头之后的数据，对齐以此为起点计算
    buf: &'a [u8],
    pos: usize,
    little_endian: bool,
}

impl<'a> CdrReader<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        ensure!(data.len() >= 4, "CDR 数据过短");
        let little_endian = match data[1] {
            0x00 => false,
            0x01 => true,
            other => bail!("不支持的 CDR 封装类型 0x{other:02x}"),
        };
        Ok(Self { buf: &data[4..], pos: 0, little_endian })
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .context("CDR 数据越界")?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// 按 size 对齐后读取 N 字节，并按字节序转成小端排列
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.pos = self.pos.next_multiple_of(N);
        let mut bytes: [u8; N] = self.take(N)?.try_into().unwrap();
        if !self.little_endian {
            bytes.reverse();
        }
        Ok(bytes)
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_array()?))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    /// 读取 n 个字节（无对齐），用于 uint8[] 等字节序列
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    pub fn read_string(&mut self) -> Result<&'a str> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        let bytes = bytes.strip_suffix(&[0]).unwrap_or(bytes);
        std::str::from_utf8(bytes).context("字符串不是合法的 UTF-8")
    }

    /// 读取 std_msgs/msg/Header，返回 stamp（ns）
    pub fn read_header(&mut self) -> Result<i64> {
        let sec = self.read_i32()?;
        let nanosec = self.read_u32()?;
        self.read_string()?; // frame_id
        Ok(sec as i64 * 1_000_000_000 + nanosec as i64)
    }
}
