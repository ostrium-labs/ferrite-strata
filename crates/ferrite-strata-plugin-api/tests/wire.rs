//! The wire format, exercised through its public API.
//!
//! Two halves. The first half says a subgraph and a capability set survive the
//! round trip byte for byte, including every op variant and every attribute type
//! — because a codec that loses `Softmax`'s axis is a codec that produces a
//! numerically wrong answer rather than an error.
//!
//! The second half is the part that matters more, because the input is a plugin.
//! Every hostile case here is a blob a wrong or malicious plugin could send, and
//! the property under test is the same in each: **the blob is rejected, and
//! nothing large is allocated on the way to finding out**.

use ferrite_strata::attrs::{AttrKey, AttrValue, Attributes};
use ferrite_strata::capabilities::{
    CapabilitySet, ComputeUnit, DTypeMask, DeclaredLimits, PatternSupport, SupportLevel,
};
use ferrite_strata::dtype::{DType, ElementType, QuantSpec};
use ferrite_strata::graph::{Graph, GraphBuilder, Op, Value};
use ferrite_strata::id::ValueId;
use ferrite_strata::shape::{Shape, ShapeClass};

use ferrite_strata_plugin_api::raw::{StrataElementType, StrataOpKind};
use ferrite_strata_plugin_api::wire::{self, SectionKind, WireError};

/// A graph using one of every op variant, so the codec has no untested arm.
fn every_op() -> Graph {
    use ferrite_strata::id::ValueId;

    let f32 = DType::plain(ElementType::F32);
    let m23 = (f32, Shape::new(&[2, 3]).expect("within the rank limit"));
    let m24 = (f32, Shape::new(&[2, 4]).expect("within the rank limit"));
    let m32 = (f32, Shape::new(&[3, 2]).expect("within the rank limit"));
    let m22 = (f32, Shape::new(&[2, 2]).expect("within the rank limit"));
    let b123 = (f32, Shape::new(&[1, 2, 3]).expect("within the rank limit"));

    let mut b = GraphBuilder::new("every-op");
    let x = b
        .input(
            f32,
            Shape::new(&[2, 3]).expect("within the rank limit"),
            "x",
        )
        .expect("a valid input");
    let y = b
        .input(
            f32,
            Shape::new(&[2, 3]).expect("within the rank limit"),
            "y",
        )
        .expect("a valid input");
    let w = b
        .input(
            f32,
            Shape::new(&[3, 4]).expect("within the rank limit"),
            "w",
        )
        .expect("a valid input");
    let bias = b
        .input(
            f32,
            Shape::vector(4).expect("within the rank limit"),
            "bias",
        )
        .expect("a valid input");
    let gamma = b
        .input(
            f32,
            Shape::vector(3).expect("within the rank limit"),
            "gamma",
        )
        .expect("a valid input");
    let beta = b
        .input(
            f32,
            Shape::vector(3).expect("within the rank limit"),
            "beta",
        )
        .expect("a valid input");

    // One output per node, so every op's payload is exercised on the way through.
    let one = |b: &mut GraphBuilder, op, inputs: &[ValueId], spec, name: &str| -> ValueId {
        b.node(op, inputs, &[spec], Attributes::new(), name)
            .expect("a well-formed node")[0]
    };

    let add = one(&mut b, Op::Add, &[x, y], m23, "add");
    let mul = one(&mut b, Op::Mul, &[x, y], m23, "mul");
    let matmul = one(&mut b, Op::MatMul, &[x, w], m24, "mm");
    let log = one(&mut b, Op::Log, &[x], m23, "log");
    let exp = one(&mut b, Op::Exp, &[x], m23, "exp");
    let softmax = one(&mut b, Op::Softmax { axis: -1 }, &[x], m23, "softmax");
    let layernorm = one(
        &mut b,
        Op::LayerNorm {
            gamma,
            beta,
            eps: 1.0e-5,
        },
        &[x],
        m23,
        "ln",
    );
    let gelu = one(&mut b, Op::Gelu, &[x], m23, "gelu");
    let linear = one(
        &mut b,
        Op::Linear {
            weight: w,
            bias: Some(bias),
        },
        &[x],
        m24,
        "linear",
    );
    let linear_nobias = one(
        &mut b,
        Op::Linear {
            weight: w,
            bias: None,
        },
        &[x],
        m24,
        "linear-nobias",
    );
    let reshape = one(
        &mut b,
        Op::Reshape {
            shape: Shape::new(&[3, 2]).expect("within the rank limit"),
        },
        &[x],
        m32,
        "reshape",
    );
    let transpose = one(
        &mut b,
        Op::Transpose {
            permutation: vec![1, 0],
        },
        &[x],
        m32,
        "transpose",
    );
    let expand = one(
        &mut b,
        Op::Expand {
            shape: Shape::new(&[1, 2, 3]).expect("within the rank limit"),
        },
        &[x],
        b123,
        "expand",
    );
    let slice = one(
        &mut b,
        Op::Slice {
            axis: 1,
            start: 0,
            end: 2,
            stride: 1,
        },
        &[x],
        m22,
        "slice",
    );

    let mut attrs = Attributes::new();
    attrs
        .set(
            AttrKey::new("alpha").expect("a valid key"),
            AttrValue::Float(0.5),
        )
        .expect("a fresh key");
    attrs
        .set(
            AttrKey::new("blocks").expect("a valid key"),
            AttrValue::Ints(vec![1, 2, 3]),
        )
        .expect("a fresh key");
    attrs
        .set(
            AttrKey::new("enabled").expect("a valid key"),
            AttrValue::Bool(true),
        )
        .expect("a fresh key");
    attrs
        .set(
            AttrKey::new("name").expect("a valid key"),
            AttrValue::Str("t".into()),
        )
        .expect("a fresh key");
    attrs
        .set(
            AttrKey::new("count").expect("a valid key"),
            AttrValue::Int(-7),
        )
        .expect("a fresh key");
    let custom = b
        .node(
            Op::Custom {
                pattern_id: "fused::attention-v3".into(),
            },
            &[matmul],
            &[m24],
            attrs,
            "custom",
        )
        .expect("a well-formed node")[0];

    for out in [
        add,
        mul,
        log,
        exp,
        softmax,
        layernorm,
        gelu,
        linear,
        linear_nobias,
        reshape,
        transpose,
        expand,
        slice,
        custom,
    ] {
        b.output(out).expect("a valid output");
    }
    b.build()
}

#[test]
fn a_graph_survives_the_round_trip() {
    let original = every_op();
    let bytes = wire::graph::encode(&original);
    let decoded = wire::graph::decode(&bytes).expect("the blob this crate wrote must decode");

    assert_eq!(
        decoded, original,
        "the decoded graph differs from the original"
    );
}

#[test]
fn every_op_variant_has_a_distinct_wire_tag() {
    let original = every_op();
    let decoded = wire::graph::decode(&wire::graph::encode(&original)).expect("round trip");

    let tags: Vec<u32> = decoded
        .nodes()
        .iter()
        .map(|n| wire::graph::op_to_raw(&n.op) as u32)
        .collect();
    let mut sorted = tags.clone();
    sorted.sort_unstable();
    sorted.dedup();
    // `every_op` builds one node per op that takes operands, so `Input` — which
    // the builder models as a value with no producer rather than as a node — is
    // the one variant absent. Fourteen distinct tags, no duplicates.
    assert_eq!(sorted.len(), 14, "tags: {sorted:?}");

    // And every tag the ABI defines, `Input` included, decodes back to itself.
    for raw in 0..=StrataOpKind::Custom as u32 {
        let kind = StrataOpKind::from_u32(raw).unwrap_or_else(|| {
            panic!("tag {raw} is assigned to no variant, so it is a dead wire value")
        });
        assert_eq!(kind as u32, raw);
    }
}

#[test]
fn every_element_type_maps_to_a_tag_and_back() {
    // The ABI's tags are hand-assigned, so this is the test that catches a core
    // variant being added without one. `ElementType::all()` is a function rather
    // than a const slice precisely so that adding a variant fails here.
    for element in ElementType::all() {
        let tag = wire::graph::element_to_raw(*element);
        let round = wire::graph::dtype_from_raw(wire::graph::dtype_to_raw(DType::plain(*element)));
        assert_eq!(
            round.element,
            *element,
            "`{}` did not survive as tag {tag:?}",
            element.name()
        );
    }
}

#[test]
fn a_quantised_dtype_survives_the_round_trip() {
    let dtype = DType::quantised(
        ElementType::I8,
        QuantSpec::Block {
            block_size: 64,
            symmetric: true,
        },
    );
    let mut b = GraphBuilder::new("quant");
    let x = b
        .input(
            dtype,
            Shape::new(&[2, 128]).expect("within the rank limit"),
            "x",
        )
        .expect("a valid input");
    b.output(x).expect("a valid output");
    let original = b.build();

    let decoded = wire::graph::decode(&wire::graph::encode(&original)).expect("round trip");
    assert_eq!(decoded, original);
    assert_eq!(decoded.value(ValueId(0)).dtype.quant.block_size(), Some(64));
}

#[test]
fn a_bare_integer_dtype_is_rejected() {
    // A raw i8 tensor has no defined numeric meaning, so a blob that claims one
    // is refused rather than reinterpreted as something a backend can act on.
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("bare-i8");
    w.u32(1); // one value
    let (element, quant, block, symmetric) =
        wire::graph::dtype_to_parts(DType::plain(ElementType::I8));
    w.u32(element);
    w.u32(quant);
    w.u32(block);
    w.u8(symmetric);
    w.u32(1);
    w.u64(8);
    w.u64(u64::MAX);
    w.u32(0);
    w.string("v0");
    w.u32(0); // no nodes
    w.u32(0); // no outputs

    match wire::graph::decode(&w.finish()) {
        Err(WireError::BareIntegerDtype { .. }) => {}
        other => panic!("expected a bare-integer rejection, got {other:?}"),
    }
}

#[test]
fn a_capability_set_survives_the_round_trip() {
    let caps = CapabilitySet::empty("acme-npu")
        .with_dtype_mask(DTypeMask::all())
        .with_pattern(PatternSupport::with_reason(
            "ref::matmul",
            DType::plain(ElementType::F32),
            ShapeClass::Matrix,
            SupportLevel::Fused,
            "tiled by 8",
        ))
        .with_pattern(PatternSupport::with_reason(
            "fused::attention-v3",
            DType::plain(ElementType::BF16),
            ShapeClass::Batch4,
            SupportLevel::Refuse,
            "no bf16 mma on this revision",
        ))
        .with_pattern(PatternSupport::new(
            "ref::gelu",
            DType::plain(ElementType::F32),
            ShapeClass::HigherRank { rank: 6 },
            SupportLevel::Fallback,
        ))
        .with_compute_unit(ComputeUnit {
            index: 0,
            name: "npu0".into(),
            vendor_kind: "Axon-4".into(),
            memory_bytes: Some(1 << 30),
        })
        .with_limits(DeclaredLimits {
            max_allocation_bytes: Some(1 << 28),
            max_graph_bytes: None,
            max_rank: Some(8),
            max_dim: None,
            max_nodes: Some(10_000),
        });

    let decoded = wire::caps::decode(&wire::caps::encode(&caps)).expect("round trip");
    assert_eq!(decoded, caps);
}

#[test]
fn an_unset_limit_is_unset_again_rather_than_max() {
    // The sentinel is `u64::MAX`. If a decoder turned that into `Some(u64::MAX)`
    // the plugin would be reported as having a limit of 2^64 bytes, which is
    // indistinguishable from having one that large and is not what was declared.
    let caps = CapabilitySet::empty("sparse").with_limits(DeclaredLimits {
        max_allocation_bytes: None,
        max_graph_bytes: Some(4096),
        max_rank: None,
        max_dim: None,
        max_nodes: None,
    });
    let decoded = wire::caps::decode(&wire::caps::encode(&caps)).expect("round trip");
    assert_eq!(decoded.limits.max_allocation_bytes, None);
    assert_eq!(decoded.limits.max_graph_bytes, Some(4096));
    assert_eq!(decoded.limits.max_rank, None);
    assert_eq!(decoded.limits.max_nodes, None);
}

#[test]
fn a_blob_carries_its_sections_and_skips_ones_it_cannot_read() {
    let graph = every_op();
    let caps = CapabilitySet::empty("acme-npu");

    let mut blob = wire::Blob::new();
    blob.push(SectionKind::Graph, &wire::graph::encode(&graph));
    blob.push(SectionKind::Capabilities, &wire::caps::encode(&caps));
    let bytes = blob.finish();

    let parsed = wire::Parsed::parse(&bytes).expect("the blob this crate wrote must parse");
    assert_eq!(parsed.major, wire::FORMAT_MAJOR);
    let section = parsed.section(SectionKind::Graph).expect("a graph section");
    assert_eq!(
        wire::graph::decode(section).expect("decode"),
        graph,
        "a graph that went through the framing must be unchanged"
    );
}

#[test]
fn an_unknown_section_kind_is_skipped_rather_than_fatal() {
    // The forward-compatibility case: a 0.2 plugin adds a section a 0.1 host has
    // never heard of, and the host reads the sections it does know.
    let graph = every_op();
    let mut blob = wire::Blob::new();
    blob.push(SectionKind::Graph, &wire::graph::encode(&graph));
    blob.push(SectionKind::Capabilities, &[0xde, 0xad, 0xbe, 0xef]);

    // Claim the section is a kind nobody has assigned.
    let mut bytes = blob.finish();
    // The table follows the header and a four-byte section count, so the second
    // entry starts one entry-width past the first.
    let second_entry = wire::HEADER_LEN + 4 + 20;
    bytes[second_entry..second_entry + 4].copy_from_slice(&99u32.to_le_bytes());

    let parsed = wire::Parsed::parse(&bytes).expect("an unknown section must not fail the parse");
    assert!(
        parsed.section(SectionKind::Capabilities).is_none(),
        "the unknown section must not be handed out as a known one"
    );
    let section = parsed
        .section(SectionKind::Graph)
        .expect("the graph section survives");
    assert_eq!(wire::graph::decode(section).expect("decode"), graph);
}

#[test]
fn a_blob_that_is_not_a_blob_is_rejected_on_its_first_eight_bytes() {
    let bytes = b"not a strata blob at all, but long enough to have a header";
    match wire::Parsed::parse(bytes) {
        Err(WireError::BadMagic { .. }) => {}
        other => panic!("expected a bad-magic rejection, got {other:?}"),
    }
}

#[test]
fn a_wrong_format_major_is_refused_and_a_higher_minor_is_not() {
    let graph = every_op();
    let mut blob = wire::Blob::new();
    blob.push(SectionKind::Graph, &wire::graph::encode(&graph));
    let good = blob.finish();

    // A major from the future.
    let mut future = good.clone();
    future[8..12].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        wire::Parsed::parse(&future),
        Err(WireError::UnsupportedFormat { major: 99, .. })
    ));

    // A minor from the future: every field this build knows is at the same
    // offset, so the blob is readable and the graph decodes.
    let mut newer = good.clone();
    newer[12..16].copy_from_slice(&(wire::FORMAT_MINOR + 7).to_le_bytes());
    let parsed = wire::Parsed::parse(&newer).expect("a higher minor is readable");
    let section = parsed.section(SectionKind::Graph).expect("a graph section");
    assert_eq!(wire::graph::decode(section).expect("decode"), graph);
}

#[test]
fn a_truncated_blob_is_refused_rather_than_read_past_the_end() {
    let graph = every_op();
    let mut blob = wire::Blob::new();
    blob.push(SectionKind::Graph, &wire::graph::encode(&graph));
    let bytes = blob.finish();

    for cut in [1, 4, 20, bytes.len() / 2, bytes.len() - 1] {
        let truncated = &bytes[..cut];
        let result = wire::Parsed::parse(truncated).and_then(|parsed| {
            parsed
                .section(SectionKind::Graph)
                .map_or(Ok(()), |section| wire::graph::decode(section).map(|_| ()))
        });
        assert!(result.is_err(), "a blob cut to {cut} bytes must not decode");
    }
}

#[test]
fn a_length_field_larger_than_the_payload_is_refused_before_allocating() {
    // The property: a four-byte count in an untrusted blob must not be able to
    // ask for a four-gigabyte allocation. `Vec::with_capacity` would happily try,
    // so the bounds check has to happen first and it has to be the check that is
    // exercised here.
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("hostile");
    w.u32(u32::MAX); // the value count

    match wire::graph::decode(&w.finish()) {
        Err(WireError::LengthOverflow { .. }) => {}
        other => panic!("expected a length-overflow rejection, got {other:?}"),
    }
}

#[test]
fn a_string_length_larger_than_the_payload_is_refused() {
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.u32(u32::MAX); // the graph name's length
    match wire::graph::decode(&w.finish()) {
        Err(WireError::LengthOverflow { .. }) => {}
        other => panic!("expected a length-overflow rejection, got {other:?}"),
    }
}

#[test]
fn a_string_that_is_not_utf8_is_refused() {
    // Strings end up in logs and in capability tables a human reads, so invalid
    // UTF-8 in one is a way to smuggle terminal escapes into a developer's
    // terminal. It is rejected rather than lossily converted.
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.u32(2);
    w.bytes(&[0xff, 0xfe]);
    match wire::graph::decode(&w.finish()) {
        Err(WireError::BadUtf8) => {}
        other => panic!("expected a bad-UTF-8 rejection, got {other:?}"),
    }
}

#[test]
fn an_unknown_op_tag_is_refused_rather_than_guessed() {
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("future-op");
    w.u32(0); // no values
    w.u32(1); // one node
    w.u32(4242); // a tag nobody assigned
    w.u32(0);
    w.u32(0);
    w.u32(0);
    w.string("n0");
    w.u32(0);
    match wire::graph::decode(&w.finish()) {
        Err(WireError::UnknownTag {
            field: "op",
            tag: 4242,
        }) => {}
        other => panic!("expected an unknown-op-tag rejection, got {other:?}"),
    }
}

#[test]
fn a_node_count_larger_than_the_payload_is_refused() {
    // The same bounds check, on the other arena. A graph's node count is what a
    // partitioner would allocate `Vec<Node>` from.
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("hostile-nodes");
    w.u32(0); // no values
    w.u32(u32::MAX); // one node, honestly
    match wire::graph::decode(&w.finish()) {
        Err(WireError::LengthOverflow { .. }) => {}
        other => panic!("expected a length-overflow rejection, got {other:?}"),
    }
}

#[test]
fn a_node_reading_a_value_nothing_has_produced_is_refused() {
    // Every field is well-typed; the graph is still nonsense. Handing that to a
    // backend would be handing it something no backend could have produced, and
    // the read it describes is a use-before-definition rather than a cycle.
    let f32 = DType::plain(ElementType::F32);
    let spec = (f32, Shape::new(&[4]).expect("within the rank limit"));

    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("out-of-order");
    w.u32(2); // two values
    for (name, producer) in [("a", None), ("b", Some(1u32))] {
        let (element, quant, block, symmetric) = wire::graph::dtype_to_parts(f32);
        w.u32(element);
        w.u32(quant);
        w.u32(block);
        w.u8(symmetric);
        w.u32(1);
        w.u64(4);
        // `b` claims node 1 produced it, so it is not a graph input.
        match producer {
            None => w.u64(u64::MAX),
            Some(node) => w.u64(node as u64 | (1 << 63)),
        }
        w.u32(0);
        w.string(name);
    }

    w.u32(2); // two nodes
    // n0 reads v1, which n1 has not produced yet.
    w.u32(StrataOpKind::Add as u32);
    w.u32(1);
    w.u32(1);
    w.u32(1);
    w.u32(1); // n0's own output slot 1
    w.u32(0); // no attributes
    w.string("n0");
    // n1 produces v1 from v0.
    w.u32(StrataOpKind::Add as u32);
    w.u32(1);
    w.u32(0);
    w.u32(1);
    w.u32(1);
    w.u32(0);
    w.string("n1");

    let _ = spec;
    w.u32(0); // no graph outputs
    match wire::graph::decode(&w.finish()) {
        Err(WireError::Graph(_)) => {}
        other => panic!("expected a graph validation failure, got {other:?}"),
    }
}

#[test]
fn a_value_claiming_a_node_that_does_not_exist_is_refused() {
    let f32 = DType::plain(ElementType::F32);
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("phantom-producer");
    w.u32(1);
    let (element, quant, block, symmetric) = wire::graph::dtype_to_parts(f32);
    w.u32(element);
    w.u32(quant);
    w.u32(block);
    w.u8(symmetric);
    w.u32(1);
    w.u64(4);
    w.u64(99 | (1 << 63)); // claims node 99 produced it
    w.u32(0);
    w.string("v0");
    w.u32(0); // no nodes at all
    w.u32(0); // no outputs
    match wire::graph::decode(&w.finish()) {
        Err(WireError::Graph(_)) => {}
        other => panic!("expected a graph validation failure, got {other:?}"),
    }
}

#[test]
fn a_shape_beyond_the_maximum_rank_is_refused() {
    let mut w = wire::Writer::new();
    w.u32(wire::graph::schema());
    w.string("rank-9");
    w.u32(1); // one value
    let (element, quant, block, symmetric) =
        wire::graph::dtype_to_parts(DType::plain(ElementType::F32));
    w.u32(element);
    w.u32(quant);
    w.u32(block);
    w.u8(symmetric);
    w.u32(9); // rank 9, one past the maximum
    for _ in 0..9 {
        w.u64(2);
    }
    w.u64(u64::MAX);
    w.u32(0);
    w.string("v0");
    match wire::graph::decode(&w.finish()) {
        Err(WireError::Shape(_)) => {}
        other => panic!("expected a shape rejection, got {other:?}"),
    }
}

#[test]
fn every_element_type_has_an_explicit_wire_tag() {
    // Guards the hand-assigned table: `element_to_raw` falls back to F32 for a
    // variant it has no tag for, so this is the only thing that notices.
    let mapped: Vec<StrataElementType> = ElementType::all()
        .iter()
        .map(|e| wire::graph::element_to_raw(*e))
        .collect();
    let mut sorted = mapped.clone();
    sorted.sort_by_key(|e| *e as u32);
    sorted.dedup_by_key(|e| *e as u32);
    assert_eq!(
        sorted.len(),
        mapped.len(),
        "two element types share a wire tag, so one silently decodes as the other"
    );
}

#[test]
fn a_graph_named_oddly_survives() {
    // Names are the subgraph cache key, and a vendor's pattern ids end up in logs,
    // so the codec must not be tempted to normalise anything.
    let mut b = GraphBuilder::new("graph with spaces, unicode é, and \"quotes\"");
    let x = b
        .input(
            DType::plain(ElementType::F32),
            Shape::new(&[1]).expect("within the rank limit"),
            "in put",
        )
        .expect("a valid input");
    b.output(x).expect("a valid output");
    let original = b.build();
    let decoded = wire::graph::decode(&wire::graph::encode(&original)).expect("round trip");
    assert_eq!(decoded.name(), original.name());
    assert_eq!(decoded, original);
}

#[test]
fn an_empty_graph_survives() {
    let original = Graph::named("empty");
    let decoded = wire::graph::decode(&wire::graph::encode(&original)).expect("round trip");
    assert_eq!(decoded, original);
    assert_eq!(decoded.node_count(), 0);
}

#[test]
fn a_value_and_a_node_cannot_share_a_name() {
    // Checked by `Graph::from_parts` on the way out, so a blob cannot smuggle in
    // a graph whose cache key is ambiguous.
    let clash = Graph::from_parts(
        vec![ferrite_strata::graph::Node {
            op: Op::Gelu,
            inputs: vec![],
            outputs: vec![],
            attrs: Attributes::new(),
            name: ferrite_strata::id::StableName::new("same"),
        }],
        vec![Value {
            dtype: DType::plain(ElementType::F32),
            shape: Shape::new(&[1]).expect("within the rank limit"),
            producer: None,
            output_slot: 0,
            name: ferrite_strata::id::StableName::new("same"),
        }],
        vec![],
        "clash",
    );
    assert!(matches!(
        clash,
        Err(ferrite_strata::graph::GraphError::DuplicateName { .. })
    ));
}
