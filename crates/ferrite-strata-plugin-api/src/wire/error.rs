//! Errors from reading or writing a wire blob.
//!
//! Every variant is something a hostile or merely buggy plugin can cause, and
//! none of them are `Internal`: the host did not break, the *input* did. The one
//! that matters most is [`WireError::Truncated`], because the alternative —
//! trusting a length field in a blob and allocating from it — turns a malformed
//! subgraph into an out-of-bounds read inside the host.

use core::fmt;
use ferrite_strata::attrs::AttrError;
use ferrite_strata::graph::GraphError;
use ferrite_strata::shape::ShapeError;

/// Something wrong with a wire blob.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum WireError {
    /// The payload ended in the middle of a field.
    Truncated {
        /// How many more bytes the field needed.
        needed: usize,
        /// How many were left.
        available: usize,
    },
    /// The blob is not a Strata wire blob at all.
    BadMagic {
        /// What the header said.
        got: u64,
    },
    /// The blob's format is one this build cannot read.
    ///
    /// A major mismatch, or a minor *below* the one this build needs: an unknown
    /// higher minor is readable, because every field this build knows about is
    /// still at the same offset.
    UnsupportedFormat {
        /// The blob's format major.
        major: u32,
        /// The blob's format minor.
        minor: u32,
    },
    /// A tag this build has never heard of, in a field where guessing is unsafe.
    UnknownTag {
        /// What kind of field: `"element-type"`, `"op"`, `"attr"`, `"section"`.
        field: &'static str,
        /// The tag value.
        tag: u32,
    },
    /// A string field was not valid UTF-8.
    ///
    /// Checked rather than assumed because pattern ids and backend names reach a
    /// log, and invalid UTF-8 in a log is a way to smuggle terminal escapes into
    /// a developer's terminal.
    BadUtf8,
    /// A length field was larger than the remaining payload.
    ///
    /// Rejected *before* any allocation. This is the check that keeps a
    /// four-byte `u32` in an untrusted blob from asking for four gigabytes.
    LengthOverflow {
        /// The declared length.
        declared: u64,
        /// What was actually left.
        available: u64,
    },
    /// The graph inside the blob is not well-formed.
    Graph(GraphError),
    /// A shape inside the blob is not well-formed.
    Shape(ShapeError),
    /// An attribute key inside the blob was rejected by the core crate.
    Attr(AttrError),
    /// An integer element type arrived with no scaling convention.
    ///
    /// Its own variant rather than a graph error because it is the one thing a
    /// plugin can put in a blob that every field-level check passes and that no
    /// backend can act on: a bare `i8` tensor is eight bits with no defined
    /// numeric meaning, and a vendor that scaled it once already has a model
    /// that silently produces wrong numbers.
    BareIntegerDtype {
        /// The offending element type's name.
        element: &'static str,
    },
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, available } => write!(
                f,
                "the blob ended mid-field: needed {needed} more bytes, {available} available"
            ),
            Self::BadMagic { got } => {
                write!(f, "not a Strata wire blob: magic was {got:#018x}")
            }
            Self::UnsupportedFormat { major, minor } => write!(
                f,
                "wire format {major}.{minor} is not readable by this build, which \
                 speaks {}.{}",
                super::FORMAT_MAJOR,
                super::FORMAT_MINOR
            ),
            Self::UnknownTag { field, tag } => {
                write!(f, "unknown {field} tag {tag} in a blob from a newer plugin")
            }
            Self::BadUtf8 => f.write_str("a string field was not valid UTF-8"),
            Self::LengthOverflow {
                declared,
                available,
            } => write!(
                f,
                "a length field declared {declared} bytes with {available} left in \
                 the payload"
            ),
            Self::Graph(err) => write!(f, "the graph in the blob is malformed: {err}"),
            Self::Shape(err) => write!(f, "a shape in the blob is malformed: {err}"),
            Self::Attr(err) => write!(f, "an attribute key in the blob is invalid: {err}"),
            Self::BareIntegerDtype { element } => write!(
                f,
                "the blob describes an `{element}` tensor with no scaling convention. \
                 A bare integer tensor has no defined numeric meaning, so it is \
                 rejected rather than reinterpreted"
            ),
        }
    }
}

impl core::error::Error for WireError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Graph(err) => Some(err),
            Self::Shape(err) => Some(err),
            Self::Attr(err) => Some(err),
            _ => None,
        }
    }
}
