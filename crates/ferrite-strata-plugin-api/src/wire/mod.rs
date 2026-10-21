//! The versioned binary format that crosses the trust boundary.
//!
//! # Why this is hand-rolled
//!
//! The obvious choice is `serde` plus a binary codec, and it is the wrong one
//! here. `serde` derives are tied to the Rust *struct layout* of the types they
//! serialise, so the wire format changes whenever the structs change — adding a
//! field to `Node` is a semver-minor bump for us and a *silent misread* for
//! every plugin built against the previous release. `burn-ir` has exactly this
//! problem: plain serde plus ciborium with no schema versioning at all, which is
//! listed in
//! [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md)
//! as one of the things to avoid.
//!
//! So the format is explicit: every tag is a number this file assigns, every
//! length is checked before it is trusted, and every field that might grow is
//! either length-prefixed or in a section that can be skipped.
//!
//! # The framing
//!
//! ```text
//! +---------------------------+  magic u64
//! | 0x5354_5241_5441_4731     |
//! +---------------------------+  format_major u32
//! | format major              |
//! +---------------------------+  format_minor u32
//! | format minor              |
//! +---------------------------+  payload_len u64
//! | payload length            |
//! +---------------------------+
//! | section count             |  u32
//! +---------------------------+
//! | section table             |  count * { kind u32, offset u64, len u64 }
//! +---------------------------+
//! | ... unclaimed space ...   |
//! +---------------------------+
//! | section payloads          |
//! +---------------------------+
//! ```
//!
//! A **section table** rather than a bare payload, for one reason: forward
//! compatibility. A 0.2 plugin can add a section this build has never heard of,
//! and a 0.1 host skips it by its declared length and reads the sections it does
//! know. Without the table, every new field would be an offset change and a
//! minor version bump would be a major one.
//!
//! # Untrusted input
//!
//! Everything here reads bytes a plugin produced. Two rules follow, and both are
//! load-bearing rather than stylistic:
//!
//! 1. **No length is trusted before it is bounds-checked.** A `u32` count from a
//!    blob is compared against the remaining payload *before* any `Vec::with_capacity`.
//!    Getting this wrong is an attacker-chosen allocation of up to 4 GiB from a
//!    four-byte header.
//! 2. **Strings are UTF-8 validated.** They end up in logs and in capability
//!    tables a human reads.

pub mod caps;
mod error;
pub mod graph;

pub use error::WireError;

use crate::raw::{MAGIC, StrataWireBlob};

/// The wire format this build writes.
pub const FORMAT_MAJOR: u32 = 0;

/// The wire format this build's minor revision.
pub const FORMAT_MINOR: u32 = 1;

/// `b"STRATAG1"`, little-endian.
pub const MAGIC_BYTES: [u8; 8] = *b"STRATAG1";

/// The header length in bytes: magic, major, minor, payload length.
pub const HEADER_LEN: usize = 8 + 4 + 4 + 8;

/// What a section holds.
///
/// Open by design: a number this build does not know is *skipped*, because the
/// whole point of the table is that a newer plugin can add one.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SectionKind {
    /// A graph, encoded by [`graph::encode`].
    Graph = 1,
    /// A capability set, encoded by [`caps::encode`].
    Capabilities = 2,
}

impl SectionKind {
    /// Decode a section tag, or `None` for one this build does not know.
    #[must_use]
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            1 => Some(Self::Graph),
            2 => Some(Self::Capabilities),
            _ => None,
        }
    }
}

/// A blob under construction.
///
/// Sections are written as they are added and the table is patched at
/// [`Blob::finish`](Blob::finish), because a section's offset is not known until
/// the ones before it are sized.
#[derive(Clone, Debug, Default)]
pub struct Blob {
    sections: Vec<(SectionKind, u32, u64)>,
    payload: Vec<u8>,
}

impl Blob {
    /// An empty blob.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a section, returning its index.
    pub fn push(&mut self, kind: SectionKind, bytes: &[u8]) -> u32 {
        // The table sits between the header and the payloads, so every offset is
        // relative to the end of the header *plus* the size the table will be
        // once this section is in it. Reserving a u64 for every section up front
        // is not possible (the count grows), so offsets are stored relative to
        // the header and corrected at finish time.
        let index = u32::try_from(self.sections.len())
            .expect("a blob with 2^32 sections is not addressable");
        let offset = u64::try_from(self.payload.len())
            .expect("a payload larger than u64 is not addressable");
        self.sections.push((kind, index, offset));
        self.payload.extend_from_slice(bytes);
        index
    }
    /// Finish into a header plus payload.
    ///
    /// The offsets in the table are made absolute here, which is the only place
    /// that has to know the table's own size.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        let count = u32::try_from(self.sections.len()).expect("section count fits u32");
        let table_bytes = usize::try_from(count)
            .expect("count fits")
            .checked_mul(SECTION_ENTRY_LEN)
            .expect("table length fits usize");
        // Section offsets are recorded relative to the start of the section data,
        // which sits after the count and the table. Both are known only here.
        let data_base = u64::try_from(COUNT_LEN + table_bytes).expect("fits u64");

        // The declared payload covers the count, the table and the data, because
        // that is the whole region a reader slices out after the header. Declaring
        // only the data would make every offset past the header wrong by exactly
        // the table's size.
        let payload_len = data_base + u64::try_from(self.payload.len()).expect("payload fits u64");

        let mut out = Vec::with_capacity(HEADER_LEN + payload_len as usize);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&FORMAT_MAJOR.to_le_bytes());
        out.extend_from_slice(&FORMAT_MINOR.to_le_bytes());
        out.extend_from_slice(&payload_len.to_le_bytes());

        out.extend_from_slice(&count.to_le_bytes());
        for (kind, _, offset) in &self.sections {
            out.extend_from_slice(&kind.as_u32().to_le_bytes());
            out.extend_from_slice(&(data_base + offset).to_le_bytes());
            out.extend_from_slice(&self.section_len(*offset).to_le_bytes());
        }

        out.extend_from_slice(&self.payload);
        out
    }

    /// Finish into the `#[repr(C)]` header plus a borrowed payload pointer.
    ///
    /// The returned `Vec` owns the bytes; the header borrows from it, so the
    /// caller must keep both alive for as long as the pointer is used.
    pub fn finish_borrowed(self) -> (StrataWireBlob, Vec<u8>) {
        let bytes = self.finish();
        let payload = bytes[HEADER_LEN..].to_vec();
        let header = StrataWireBlob {
            magic: MAGIC,
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            payload_len: payload.len() as u64,
            data: payload.as_ptr(),
            nul: 0,
        };
        (header, payload)
    }

    fn section_len(&self, offset: u64) -> u64 {
        let start = offset as usize;
        let next = self
            .sections
            .iter()
            .map(|(_, _, off)| *off as usize)
            .filter(|&off| off > start)
            .min()
            .unwrap_or(self.payload.len());
        (next - start) as u64
    }
}

impl SectionKind {
    const fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Bytes per table entry: kind, offset, length.
const SECTION_ENTRY_LEN: usize = 4 + 8 + 8;

/// Bytes for the section count that precedes the table.
const COUNT_LEN: usize = 4;

/// A parsed wire blob: the header, plus whatever sections it had.
#[derive(Clone, Debug)]
pub struct Parsed<'a> {
    /// The declared format major.
    pub major: u32,
    /// The declared format minor.
    pub minor: u32,
    raw: &'a [u8],
    sections: Vec<(u32, &'a [u8])>,
}

impl<'a> Parsed<'a> {
    /// Parse a blob's header and section table.
    ///
    /// Sections whose kind is unknown are dropped here rather than reported: an
    /// unreadable section is not an error, it is the forward-compatibility case.
    ///
    /// # Errors
    ///
    /// [`WireError::BadMagic`] if this is not a wire blob,
    /// [`WireError::UnsupportedFormat`] if the major is not
    /// [`FORMAT_MAJOR`], or [`WireError::Truncated`] if the table does not fit
    /// in what the header says is there.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, WireError> {
        let mut reader = Reader::new(bytes);
        let magic = reader.u64()?;
        if magic != MAGIC {
            return Err(WireError::BadMagic { got: magic });
        }
        let major = reader.u32()?;
        let minor = reader.u32()?;
        let payload_len = reader.u64()?;
        if major != FORMAT_MAJOR {
            return Err(WireError::UnsupportedFormat { major, minor });
        }

        let declared = usize::try_from(payload_len).map_err(|_| WireError::Truncated {
            needed: payload_len as usize,
            available: bytes.len().saturating_sub(HEADER_LEN),
        })?;
        let available = bytes.len().saturating_sub(HEADER_LEN);
        if declared > available {
            return Err(WireError::Truncated {
                needed: declared,
                available,
            });
        }
        let payload = &bytes[HEADER_LEN..HEADER_LEN + declared];

        let mut table = Reader::new(payload);
        let count = table.u32()?;
        let mut sections = Vec::with_capacity(
            usize::try_from(count)
                .ok()
                .filter(|&n| n <= table.remaining() / SECTION_ENTRY_LEN)
                .unwrap_or(0),
        );
        for _ in 0..count {
            if table.remaining() < SECTION_ENTRY_LEN {
                break;
            }
            let kind = table.u32()?;
            let offset = table.u64()?;
            let len = table.u64()?;
            let start = usize::try_from(offset).unwrap_or(usize::MAX);
            let size = usize::try_from(len).unwrap_or(usize::MAX);
            // Both halves bounds-checked before the slice, and an out-of-range
            // section is skipped rather than fatal: a newer plugin's section may
            // legitimately be past anything this build reasoned about.
            if start.saturating_add(size) <= payload.len() && SectionKind::from_u32(kind).is_some()
            {
                sections.push((kind, &payload[start..start + size]));
            }
        }

        Ok(Self {
            major,
            minor,
            raw: payload,
            sections,
        })
    }

    /// The section of a kind this build knows, if present.
    #[must_use]
    pub fn section(&self, kind: SectionKind) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|(k, _)| *k == kind.as_u32())
            .map(|(_, bytes)| *bytes)
    }

    /// Every section kind present, including ones this build does not know.
    #[must_use]
    pub fn kinds(&self) -> Vec<u32> {
        self.sections.iter().map(|(kind, _)| *kind).collect()
    }

    /// The whole payload, for a caller that wants to hash it.
    #[must_use]
    pub fn payload(&self) -> &'a [u8] {
        self.raw
    }
}

/// A bounds-checked little-endian reader.
#[derive(Clone, Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    /// A reader over `bytes`.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// How many bytes are left.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }

    /// Take exactly `n` bytes.
    ///
    /// # Errors
    ///
    /// [`WireError::Truncated`] if fewer than `n` bytes remain. Every other read
    /// goes through here, so this is the single place a short blob is caught.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.at.checked_add(n).ok_or(WireError::Truncated {
            needed: n,
            available: self.remaining(),
        })?;
        if end > self.bytes.len() {
            return Err(WireError::Truncated {
                needed: n,
                available: self.remaining(),
            });
        }
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    /// Read a `u8`.
    ///
    /// # Errors
    ///
    /// [`WireError::Truncated`] if the reader is at the end.
    pub fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    /// Read a `u32`, little-endian.
    ///
    /// # Errors
    ///
    /// [`WireError::Truncated`] if fewer than four bytes remain.
    pub fn u32(&mut self) -> Result<u32, WireError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("take(4) yields four bytes"),
        ))
    }

    /// Read a `u64`, little-endian.
    ///
    /// # Errors
    ///
    /// [`WireError::Truncated`] if fewer than eight bytes remain.
    pub fn u64(&mut self) -> Result<u64, WireError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("take(8) yields eight bytes"),
        ))
    }

    /// Read a length-prefixed string, checked against the bytes that remain.
    ///
    /// # Errors
    ///
    /// [`WireError::LengthOverflow`] if the declared length exceeds what is
    /// left — raised *before* any allocation — or [`WireError::BadUtf8`] if the
    /// bytes are not UTF-8.
    pub fn string(&mut self) -> Result<String, WireError> {
        let len = self.u32()? as usize;
        if len > self.remaining() {
            return Err(WireError::LengthOverflow {
                declared: len as u64,
                available: self.remaining() as u64,
            });
        }
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| WireError::BadUtf8)
    }

    /// Read a `u32` count and check it against the remaining bytes.
    ///
    /// `min_item_bytes` is the smallest number of bytes one item can occupy, and
    /// is what makes the check meaningful: a count of 2 billion with
    /// `min_item_bytes` of 16 is rejected without allocating.
    ///
    /// # Errors
    ///
    /// [`WireError::LengthOverflow`] if `count * min_item_bytes` exceeds what is
    /// left.
    pub fn count(&mut self, min_item_bytes: usize) -> Result<usize, WireError> {
        let raw = self.u32()? as usize;
        let needed = raw.saturating_mul(min_item_bytes);
        if needed > self.remaining() {
            return Err(WireError::LengthOverflow {
                declared: needed as u64,
                available: self.remaining() as u64,
            });
        }
        Ok(raw)
    }
}

/// A little-endian byte writer.
#[derive(Clone, Debug, Default)]
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Finish into a `Vec`.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }

    /// Append a `u8`.
    pub fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }

    /// Append a `u32`, little-endian.
    pub fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a `u64`, little-endian.
    pub fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a `bool` as one byte.
    pub fn bool(&mut self, v: bool) {
        self.bytes.push(u8::from(v));
    }

    /// Append a length-prefixed string.
    pub fn string(&mut self, v: &str) {
        let len = u32::try_from(v.len()).expect("a string longer than u32::MAX");
        self.bytes.extend_from_slice(&len.to_le_bytes());
        self.bytes.extend_from_slice(v.as_bytes());
    }

    /// Append raw bytes.
    pub fn bytes(&mut self, v: &[u8]) {
        self.bytes.extend_from_slice(v);
    }

    /// A reader over what has been written so far.
    ///
    /// Used by the round-trip tests to read a section back without going through
    /// the framing.
    #[must_use]
    pub fn as_reader(&self) -> Reader<'_> {
        Reader::new(&self.bytes)
    }
}
