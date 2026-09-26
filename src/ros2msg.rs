//! 解析 ros2msg 格式的 schema 文本，并按其定义把 CDR 数据动态解码为 [`Value`]

use anyhow::{Context, Result, bail, ensure};
use std::{collections::HashMap, rc::Rc};

use crate::cdr::CdrReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Bool,
    Byte,
    Char,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
    String,
}

impl Primitive {
    fn parse(name: &str) -> Option<Self> {
        // string<=N 是有界字符串，编码与 string 相同
        if name == "string" || name.starts_with("string<=") {
            return Some(Self::String);
        }
        Some(match name {
            "bool" => Self::Bool,
            "byte" => Self::Byte,
            "char" => Self::Char,
            "int8" => Self::Int8,
            "uint8" => Self::UInt8,
            "int16" => Self::Int16,
            "uint16" => Self::UInt16,
            "int32" => Self::Int32,
            "uint32" => Self::UInt32,
            "int64" => Self::Int64,
            "uint64" => Self::UInt64,
            "float32" => Self::Float32,
            "float64" => Self::Float64,
            _ => return None,
        })
    }

    fn is_byte_like(self) -> bool {
        matches!(self, Self::Byte | Self::Char | Self::UInt8 | Self::Int8)
    }
}

#[derive(Debug, Clone)]
enum BaseType {
    Primitive(Primitive),
    /// 规范化后的消息名，如 `std_msgs/Header`
    Message(String),
}

#[derive(Debug, Clone, Copy)]
enum ArrayKind {
    None,
    /// T[N]
    Fixed(usize),
    /// T[] 或 T[<=N]，编码时带长度前缀
    Sequence,
}

#[derive(Debug, Clone)]
struct Field {
    name: Rc<str>,
    base: BaseType,
    array: ArrayKind,
}

/// 一个 schema 及其依赖的全部消息定义
#[derive(Debug)]
pub struct Schema {
    root: String,
    messages: HashMap<String, Vec<Field>>,
}

/// 动态解码结果
#[derive(Debug, Clone)]
pub enum Value {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    Array(Vec<Value>),
    Struct(Vec<(Rc<str>, Value)>),
}

/// 把 `pkg/msg/Name` 规范化为 `pkg/Name`
fn normalize(name: &str) -> String {
    name.replacen("/msg/", "/", 1)
}

fn package_of(name: &str) -> &str {
    name.split('/').next().unwrap_or("")
}

impl Schema {
    /// `name` 为 schema 名（如 `sensor_msgs/msg/JointState`），`text` 为 ros2msg 定义文本
    pub fn parse(name: &str, text: &str) -> Result<Self> {
        let root = normalize(name);
        let mut messages = HashMap::new();
        let mut current = root.clone();
        let mut lines: Vec<&str> = Vec::new();

        let mut flush = |msg: &str, lines: &[&str]| -> Result<()> {
            let package = package_of(msg);
            let fields = lines
                .iter()
                .filter_map(|line| parse_field(line, package).transpose())
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("解析消息 {msg} 失败"))?;
            messages.insert(msg.to_string(), fields);
            Ok(())
        };

        for line in text.lines() {
            if line.starts_with("====") {
                flush(&current, &lines)?;
                lines.clear();
                continue;
            }
            if let Some(dep) = line.trim().strip_prefix("MSG:") {
                current = normalize(dep.trim());
                continue;
            }
            lines.push(line);
        }
        flush(&current, &lines)?;

        let schema = Self { root, messages };
        schema.check_resolved(&schema.root)?;
        Ok(schema)
    }

    fn check_resolved(&self, msg: &str) -> Result<()> {
        let fields = self.messages.get(msg).with_context(|| format!("schema 中缺少消息定义 {msg}"))?;
        for f in fields {
            if let BaseType::Message(sub) = &f.base {
                self.check_resolved(sub)?;
            }
        }
        Ok(())
    }

    pub fn decode(&self, data: &[u8]) -> Result<Value> {
        let mut r = CdrReader::new(data)?;
        self.decode_message(&self.root, &mut r)
    }

    fn decode_message(&self, msg: &str, r: &mut CdrReader) -> Result<Value> {
        let fields = &self.messages[msg];
        let mut values = Vec::with_capacity(fields.len());
        for f in fields {
            let decode_field = |r: &mut CdrReader| match f.array {
                ArrayKind::None => self.decode_base(&f.base, r),
                ArrayKind::Fixed(n) => self.decode_array(&f.base, n, r),
                ArrayKind::Sequence => {
                    let n = r.read_u32()? as usize;
                    self.decode_array(&f.base, n, r)
                }
            };
            let v = decode_field(r).with_context(|| format!("解码字段 {msg}.{} 失败", f.name))?;
            values.push((f.name.clone(), v));
        }
        Ok(Value::Struct(values))
    }

    fn decode_array(&self, base: &BaseType, n: usize, r: &mut CdrReader) -> Result<Value> {
        if let BaseType::Primitive(p) = base
            && p.is_byte_like()
        {
            return Ok(Value::Bytes(r.read_bytes(n)?.to_vec()));
        }
        // 长度来自数据本身，先粗略检查避免异常数据导致超大分配
        ensure!(n <= 1 << 24, "数组长度 {n} 异常");
        (0..n).map(|_| self.decode_base(base, r)).collect::<Result<_>>().map(Value::Array)
    }

    fn decode_base(&self, base: &BaseType, r: &mut CdrReader) -> Result<Value> {
        let p = match base {
            BaseType::Message(sub) => return self.decode_message(sub, r),
            BaseType::Primitive(p) => *p,
        };
        Ok(match p {
            Primitive::Bool => Value::Bool(r.read_u8()? != 0),
            Primitive::Byte | Primitive::Char | Primitive::UInt8 => Value::UInt(r.read_u8()? as u64),
            Primitive::Int8 => Value::Int(r.read_u8()? as i8 as i64),
            Primitive::Int16 => Value::Int(r.read_u16()? as i16 as i64),
            Primitive::UInt16 => Value::UInt(r.read_u16()? as u64),
            Primitive::Int32 => Value::Int(r.read_u32()? as i32 as i64),
            Primitive::UInt32 => Value::UInt(r.read_u32()? as u64),
            Primitive::Int64 => Value::Int(r.read_u64()? as i64),
            Primitive::UInt64 => Value::UInt(r.read_u64()?),
            Primitive::Float32 => Value::Float(r.read_f32()? as f64),
            Primitive::Float64 => Value::Float(r.read_f64()?),
            Primitive::String => Value::String(r.read_string()?.to_string()),
        })
    }
}

/// 解析一行字段定义；空行、注释、常量返回 None
fn parse_field(line: &str, package: &str) -> Result<Option<Field>> {
    let line = line.split('#').next().unwrap_or("").trim();
    if line.is_empty() {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (Some(ty), Some(name)) = (parts.next(), parts.next()) else {
        bail!("无法解析字段定义: {line}");
    };
    // 常量形如 `uint8 FOO=1`，不参与序列化
    if name.contains('=') {
        return Ok(None);
    }

    let (base, array) = match ty.find('[') {
        None => (ty, ArrayKind::None),
        Some(i) => {
            let inner = ty[i + 1..].strip_suffix(']').with_context(|| format!("数组类型格式错误: {ty}"))?;
            let array = if inner.is_empty() || inner.starts_with("<=") {
                ArrayKind::Sequence
            } else {
                ArrayKind::Fixed(inner.parse().with_context(|| format!("数组长度格式错误: {ty}"))?)
            };
            (&ty[..i], array)
        }
    };

    let base = match Primitive::parse(base) {
        Some(p) => BaseType::Primitive(p),
        None if base == "wstring" || base.starts_with("wstring<=") => bail!("暂不支持 wstring"),
        None if base == "Header" => BaseType::Message("std_msgs/Header".into()),
        None if base.contains('/') => BaseType::Message(normalize(base)),
        None => BaseType::Message(format!("{package}/{base}")),
    };
    Ok(Some(Field { name: name.into(), base, array }))
}

impl Value {
    /// 按 `a.b.c` 形式的路径取子字段
    pub fn get_path(&self, path: &str) -> Result<&Value> {
        let mut v = self;
        for key in path.split('.') {
            let Value::Struct(fields) = v else {
                bail!("路径 {path} 中的 {key} 不是结构体字段");
            };
            v = fields
                .iter()
                .find(|(name, _)| &**name == key)
                .map(|(_, v)| v)
                .with_context(|| format!("找不到字段 {key}（路径 {path}）"))?;
        }
        Ok(v)
    }

    /// 把数值、数值数组或只含数值的结构体按顺序展开为 f64，并生成每个维度的名字
    pub fn flatten_numbers(&self, prefix: &str, out: &mut Vec<f64>, names: &mut Vec<String>) -> Result<()> {
        match self {
            Value::Bool(b) => push(out, names, prefix, *b as u8 as f64),
            Value::Int(i) => push(out, names, prefix, *i as f64),
            Value::UInt(u) => push(out, names, prefix, *u as f64),
            Value::Float(f) => push(out, names, prefix, *f),
            Value::Bytes(bytes) => {
                for (i, b) in bytes.iter().enumerate() {
                    push(out, names, &format!("{prefix}[{i}]"), *b as f64);
                }
            }
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    item.flatten_numbers(&format!("{prefix}[{i}]"), out, names)?;
                }
            }
            Value::Struct(fields) => {
                for (name, v) in fields {
                    v.flatten_numbers(&format!("{prefix}.{name}"), out, names)?;
                }
            }
            Value::String(_) => bail!("字段 {prefix} 是字符串，不能转换为数值"),
        }
        Ok(())
    }

    /// 字符串数组，如 JointState 的 name
    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            Value::Array(items) => items
                .iter()
                .map(|v| match v {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }
}

fn push(out: &mut Vec<f64>, names: &mut Vec<String>, name: &str, value: f64) {
    out.push(value);
    names.push(name.to_string());
}

/// schema 的第一个字段是否为 std_msgs/Header
pub fn has_header(schema: &mcap::Schema) -> bool {
    if schema.encoding != "ros2msg" {
        return false;
    }
    let Ok(text) = std::str::from_utf8(&schema.data) else {
        return false;
    };
    let package = package_of(&schema.name);
    text.lines()
        // 依赖的子消息定义从 "====" 分隔线开始
        .take_while(|line| !line.starts_with("===="))
        .find_map(|line| parse_field(line, package).ok().flatten())
        .is_some_and(|f| matches!(f.base, BaseType::Message(m) if m == "std_msgs/Header"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const JOINT_STATE: &str = "\
# comment
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
";

    /// 手工构造一条小端 CDR 编码的 JointState
    fn joint_state_cdr() -> Vec<u8> {
        let mut b = vec![0x00, 0x01, 0x00, 0x00];
        let mut body: Vec<u8> = Vec::new();
        let align = |body: &mut Vec<u8>, n: usize| {
            while body.len() % n != 0 {
                body.push(0);
            }
        };
        let string = |body: &mut Vec<u8>, s: &str| {
            body.extend((s.len() as u32 + 1).to_le_bytes());
            body.extend(s.as_bytes());
            body.push(0);
        };
        body.extend(7i32.to_le_bytes());
        body.extend(9u32.to_le_bytes());
        string(&mut body, "base");
        align(&mut body, 4);
        body.extend(2u32.to_le_bytes());
        string(&mut body, "j1");
        align(&mut body, 4);
        string(&mut body, "j2");
        for values in [[1.5f64, -2.0], [0.0, 0.0], [0.0, 0.0]] {
            align(&mut body, 4);
            body.extend(2u32.to_le_bytes());
            align(&mut body, 8);
            for v in values {
                body.extend(v.to_le_bytes());
            }
        }
        b.extend(body);
        b
    }

    #[test]
    fn decodes_joint_state() {
        let schema = Schema::parse("sensor_msgs/msg/JointState", JOINT_STATE).unwrap();
        let value = schema.decode(&joint_state_cdr()).unwrap();
        assert!(matches!(value.get_path("header.stamp.sec").unwrap(), Value::Int(7)));
        assert_eq!(value.get_path("name").unwrap().as_string_array().unwrap(), vec!["j1", "j2"]);

        let mut out = Vec::new();
        let mut names = Vec::new();
        value.get_path("position").unwrap().flatten_numbers("position", &mut out, &mut names).unwrap();
        assert_eq!(out, vec![1.5, -2.0]);
        assert_eq!(names, vec!["position[0]", "position[1]"]);
    }

    #[test]
    fn skips_constants_and_resolves_names() {
        let text = "uint8 FOO=1\nPose pose\n====\nMSG: geometry_msgs/Pose\nfloat64 x\n";
        let schema = Schema::parse("geometry_msgs/msg/PoseStamped", text).unwrap();
        assert_eq!(schema.messages["geometry_msgs/PoseStamped"].len(), 1);
    }

    #[test]
    fn missing_dependency_fails() {
        assert!(Schema::parse("a/msg/B", "c/D field\n").is_err());
    }
}
