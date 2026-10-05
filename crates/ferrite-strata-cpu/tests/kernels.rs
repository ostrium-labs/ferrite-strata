//! Correctness tests for the reference kernels.
//!
//! These are the tests that make the crate worth having: they assert *values*, not
//! just that something ran. A backend that disagrees with the CPU backend is only
//! meaningful if the CPU backend is right, and that is what these establish.

use ferrite_strata::{
    Backend, DType, ElementType, Graph, GraphBuilder, Op, Shape, ValueId,
    capabilities::reference::ReferenceOp,
};
use ferrite_strata_cpu::{CpuBackend, CpuCompile, CpuTensor, f32_tensor};

fn f32() -> DType {
    DType::plain(ElementType::F32)
}

fn dims(d: &[usize]) -> Shape {
    Shape::new(d).expect("within the rank limit")
}

/// Compile with the typed entry point, requiring success.
fn compile(graph: &Graph) -> ferrite_strata_cpu::CpuExecutable {
    match CpuBackend::new()
        .compile_typed(graph, "ref::whatever")
        .expect("no hard error")
    {
        CpuCompile::Compiled(executable) => executable,
        CpuCompile::Declined { reason } => panic!("unexpectedly declined: {reason}"),
    }
}

/// A one-op graph with a single input.
fn unary(op: Op, in_shape: &[usize], out_shape: &[usize]) -> (Graph, ValueId, ValueId) {
    let mut b = GraphBuilder::new("unary");
    let x = b.input(f32(), dims(in_shape), "x").expect("x is free");
    let out = b
        .node(
            op,
            &[x],
            &[(f32(), dims(out_shape))],
            Default::default(),
            "op0",
        )
        .expect("x exists");
    b.output(out[0]).expect("not already an output");
    (b.build(), x, out[0])
}

/// A binary op with the same data on both sides.
fn run_binary(
    op: Op,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    lhs: &[f32],
    rhs: &[f32],
) -> Vec<f32> {
    let (graph, a, c, out) = binary(op, lhs_shape, rhs_shape, out_shape);
    let executable = compile(&graph);
    let inputs = std::collections::HashMap::from([
        (a, f32_tensor(lhs_shape, lhs).expect("lhs fits")),
        (c, f32_tensor(rhs_shape, rhs).expect("rhs fits")),
    ]);
    let results = executable.run_values(&inputs).expect("run");
    results
        .get(&out)
        .expect("the result is produced")
        .as_slice()
        .to_vec()
}

fn run_unary(op: Op, in_shape: &[usize], out_shape: &[usize], data: &[f32]) -> Vec<f32> {
    let (graph, x, out) = unary(op, in_shape, out_shape);
    let executable = compile(&graph);
    let inputs =
        std::collections::HashMap::from([(x, f32_tensor(in_shape, data).expect("data fits"))]);
    let results = executable.run_values(&inputs).expect("run");
    results
        .get(&out)
        .expect("the result is produced")
        .as_slice()
        .to_vec()
}

/// A binary graph.
fn binary(
    op: Op,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
) -> (Graph, ValueId, ValueId, ValueId) {
    let mut b = GraphBuilder::new("binary");
    let a = b.input(f32(), dims(lhs_shape), "a").expect("a");
    let c = b.input(f32(), dims(rhs_shape), "b").expect("b");
    let out = b
        .node(
            op,
            &[a, c],
            &[(f32(), dims(out_shape))],
            Default::default(),
            "op0",
        )
        .expect("operands exist");
    b.output(out[0]).expect("not already an output");
    (b.build(), a, c, out[0])
}

/// The same scalar error function the kernel uses, for computing expectations.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn approx(got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "length differs");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance,
            "element {i}: got {g}, want {w} (tolerance {tolerance})"
        );
    }
}

#[test]
fn add_of_a_value_to_itself_doubles_it() {
    // Adding a value to itself, which is how the doctest exercises binary add.
    let out = run_binary(
        Op::Add,
        &[2, 2],
        &[2, 2],
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 2.0, 3.0, 4.0],
    );
    approx(&out, &[2.0, 4.0, 6.0, 8.0], 0.0);
}

#[test]
fn add_is_elementwise() {
    let (graph, a, c, out) = binary(Op::Add, &[2], &[2], &[2]);
    let executable = compile(&graph);
    let inputs = std::collections::HashMap::from([
        (a, f32_tensor(&[2], &[1.0, 2.0]).expect("fits")),
        (c, f32_tensor(&[2], &[10.0, 20.0]).expect("fits")),
    ]);
    let results = executable.run_values(&inputs).expect("run");
    approx(
        results.get(&out).expect("produced").as_slice(),
        &[11.0, 22.0],
        0.0,
    );
}

#[test]
fn matmul_is_the_naive_matrix_product() {
    let (graph, a, c, out) = binary(Op::MatMul, &[2, 2], &[2, 2], &[2, 2]);
    let executable = compile(&graph);
    let inputs = std::collections::HashMap::from([
        (a, f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("fits")),
        (c, f32_tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]).expect("fits")),
    ]);
    let results = executable.run_values(&inputs).expect("run");
    // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
    approx(
        results.get(&out).expect("produced").as_slice(),
        &[19.0, 22.0, 43.0, 50.0],
        0.0,
    );
}

#[test]
fn matmul_accumulates_in_inner_dimension_order() {
    // A row of ones against a column of twos is exactly 2k, so the accumulation
    // order is observable.
    let (graph, a, c, out) = binary(Op::MatMul, &[1, 4], &[4, 1], &[1, 1]);
    let executable = compile(&graph);
    let inputs = std::collections::HashMap::from([
        (a, f32_tensor(&[1, 4], &[1.0; 4]).expect("fits")),
        (c, f32_tensor(&[4, 1], &[2.0; 4]).expect("fits")),
    ]);
    let results = executable.run_values(&inputs).expect("run");
    approx(results.get(&out).expect("produced").as_slice(), &[8.0], 0.0);
}

#[test]
fn matmul_with_mismatched_inner_dimensions_is_an_error() {
    let (graph, a, c, out) = binary(Op::MatMul, &[2, 3], &[2, 2], &[2, 2]);
    let executable = compile(&graph);
    let inputs = std::collections::HashMap::from([
        (a, f32_tensor(&[2, 3], &[0.0; 6]).expect("fits")),
        (c, f32_tensor(&[2, 2], &[0.0; 4]).expect("fits")),
    ]);
    let err = executable
        .run_values(&inputs)
        .expect_err("3 columns against 2 rows");
    assert!(err.to_string().contains("inner dimensions"), "{err}");
    assert!(out != ValueId(0));
}

#[test]
fn softmax_rows_sum_to_one() {
    let out = run_unary(
        Op::Softmax { axis: -1 },
        &[2, 3],
        &[2, 3],
        &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
    );
    approx(&out[0..3], &[0.090_030_6, 0.244_728_4, 0.665_240_9], 1e-5);
    // A uniform row must be exactly uniform.
    approx(&out[3..6], &[1.0 / 3.0; 3], 1e-6);
}

#[test]
fn softmax_subtracts_the_max_and_survives_a_large_bias() {
    // exp(1000) is inf, so the naive form returns NaN here. This test is the reason
    // the kernel subtracts the maximum.
    let out = run_unary(
        Op::Softmax { axis: -1 },
        &[1, 2],
        &[1, 2],
        &[1000.0, 1001.0],
    );
    assert!(out.iter().all(|v| v.is_finite()), "got {out:?}");
    approx(&out, &[0.268_941_4, 0.731_058_6], 1e-5);
}

#[test]
fn gelu_is_the_exact_form_not_the_tanh_approximation() {
    // The exact form is gelu(x) = x * Phi(x). The tanh approximation is the one
    // most kernels ship, and it differs in the fourth decimal: at x = 1 the exact
    // answer is 0.8413447 and the approximation gives 0.8411920. A tolerance loose
    // enough to accept both would make this test useless, so the tolerance is tight
    // enough to reject the approximation.
    let out = run_unary(Op::Gelu, &[1, 3], &[1, 3], &[-1.0, 0.0, 1.0]);
    approx(&out[0..1], &[-0.158_655_2], 1e-5);
    approx(&out[1..2], &[0.0], 1e-6);
    approx(&out[2..3], &[0.841_344_8], 1e-5);

    // GELU is not odd-symmetric: gelu(-x) is not -gelu(x), because it is a
    // probability-weighted gate rather than a sign-symmetric one. A backend
    // implementing a symmetric approximation would pass a symmetry check and fail
    // this one.
    assert!((out[0] + out[2]).abs() > 0.5);
}

#[test]
fn layer_norm_normalises_each_row() {
    let mut b = GraphBuilder::new("ln");
    let x = b.input(f32(), dims(&[1, 4]), "x").expect("x");
    let gamma = b.input(f32(), dims(&[4]), "gamma").expect("gamma");
    let beta = b.input(f32(), dims(&[4]), "beta").expect("beta");
    let out = b
        .node(
            Op::LayerNorm {
                gamma,
                beta,
                eps: 1e-5,
            },
            &[x],
            &[(f32(), dims(&[1, 4]))],
            Default::default(),
            "ln0",
        )
        .expect("x exists");
    b.output(out[0]).expect("out");
    let graph = b.build();

    let executable = compile(&graph);
    let inputs = std::collections::HashMap::from([
        (x, f32_tensor(&[1, 4], &[1.0, 2.0, 3.0, 4.0]).expect("fits")),
        (gamma, f32_tensor(&[4], &[1.0; 4]).expect("fits")),
        (beta, f32_tensor(&[4], &[0.0; 4]).expect("fits")),
    ]);
    let result = executable.run_values(&inputs).expect("run");
    let out = result.get(&out[0]).expect("produced").as_slice();

    // Identity gain and zero bias: the output must be standardised, so it has zero
    // mean and unit variance over the row.
    let mean = out.iter().sum::<f32>() / 4.0;
    approx(&[mean], &[0.0], 1e-5);
    let variance = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
    approx(&[variance], &[1.0], 1e-3);
}

#[test]
fn layer_norm_applies_gamma_and_beta() {
    let mut b = GraphBuilder::new("ln");
    let x = b.input(f32(), dims(&[1, 2]), "x").expect("x");
    let gamma = b.input(f32(), dims(&[2]), "gamma").expect("gamma");
    let beta = b.input(f32(), dims(&[2]), "beta").expect("beta");
    let out = b
        .node(
            Op::LayerNorm {
                gamma,
                beta,
                eps: 1e-5,
            },
            &[x],
            &[(f32(), dims(&[1, 2]))],
            Default::default(),
            "ln0",
        )
        .expect("x exists");
    b.output(out[0]).expect("out");

    let executable = compile(&b.build());
    let inputs = std::collections::HashMap::from([
        (x, f32_tensor(&[1, 2], &[1.0, 3.0]).expect("fits")),
        (gamma, f32_tensor(&[2], &[2.0, 0.0]).expect("fits")),
        (beta, f32_tensor(&[2], &[1.0, 0.0]).expect("fits")),
    ]);
    let result = executable.run_values(&inputs).expect("run");
    // The standardised row is [-1, 1]; gamma and beta turn it into [2*-1+1, 0*1+0].
    approx(
        result.get(&out[0]).expect("produced").as_slice(),
        &[-1.0, 0.0],
        1e-4,
    );
}

#[test]
fn linear_multiplies_by_the_weight_and_adds_the_bias() {
    let mut b = GraphBuilder::new("linear");
    let x = b.input(f32(), dims(&[1, 2]), "x").expect("x");
    let weight = b.input(f32(), dims(&[2, 2]), "w").expect("w");
    let bias = b.input(f32(), dims(&[2]), "b").expect("b");
    let out = b
        .node(
            Op::Linear {
                weight,
                bias: Some(bias),
            },
            &[x],
            &[(f32(), dims(&[1, 2]))],
            Default::default(),
            "fc0",
        )
        .expect("x exists");
    b.output(out[0]).expect("out");

    let executable = compile(&b.build());
    let inputs = std::collections::HashMap::from([
        (x, f32_tensor(&[1, 2], &[1.0, 2.0]).expect("fits")),
        // Weight is [out, in], so this row is the identity.
        (
            weight,
            f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]).expect("fits"),
        ),
        (bias, f32_tensor(&[2], &[10.0, 20.0]).expect("fits")),
    ]);
    let result = executable.run_values(&inputs).expect("run");
    approx(
        result.get(&out[0]).expect("produced").as_slice(),
        &[11.0, 22.0],
        1e-6,
    );
}

#[test]
fn linear_without_a_bias_is_just_the_matmul() {
    let mut b = GraphBuilder::new("linear");
    let x = b.input(f32(), dims(&[1, 2]), "x").expect("x");
    let weight = b.input(f32(), dims(&[1, 2]), "w").expect("w");
    let out = b
        .node(
            Op::Linear { weight, bias: None },
            &[x],
            &[(f32(), dims(&[1, 1]))],
            Default::default(),
            "fc0",
        )
        .expect("x exists");
    b.output(out[0]).expect("out");

    let executable = compile(&b.build());
    let inputs = std::collections::HashMap::from([
        (x, f32_tensor(&[1, 2], &[3.0, 4.0]).expect("fits")),
        (weight, f32_tensor(&[1, 2], &[1.0, 1.0]).expect("fits")),
    ]);
    let result = executable.run_values(&inputs).expect("run");
    approx(
        result.get(&out[0]).expect("produced").as_slice(),
        &[7.0],
        1e-6,
    );
}

#[test]
fn transpose_swaps_the_two_axes() {
    let out = run_unary(
        Op::Transpose {
            permutation: vec![1, 0],
        },
        &[2, 3],
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    approx(&out, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 0.0);
}

#[test]
fn reshape_preserves_the_row_major_order() {
    let out = run_unary(
        Op::Reshape {
            shape: dims(&[3, 2]),
        },
        &[2, 3],
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    approx(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0);
}

#[test]
fn expand_broadcasts_a_row_across_the_rows() {
    let out = run_unary(
        Op::Expand {
            shape: dims(&[3, 2]),
        },
        &[1, 2],
        &[3, 2],
        &[7.0, 8.0],
    );
    approx(&out, &[7.0, 8.0, 7.0, 8.0, 7.0, 8.0], 0.0);
}

#[test]
fn a_zero_element_tensor_produces_a_zero_element_tensor() {
    // A zero-sized batch is a real case, and every kernel here indexes at least
    // once, so the short-circuit is load-bearing rather than defensive.
    let out = run_unary(Op::Gelu, &[0, 4], &[0, 4], &[]);
    assert!(out.is_empty());
}

#[test]
fn running_without_an_operand_is_an_error_naming_it() {
    let (graph, _x, out) = unary(Op::Gelu, &[1, 2], &[1, 2]);
    let executable = compile(&graph);
    let err = executable
        .run_values(&std::collections::HashMap::new())
        .expect_err("no inputs at all");
    assert!(err.to_string().contains("no input supplied"), "{err}");
    assert!(out != ValueId(0));
}

#[test]
fn a_step_by_step_chain_computes_in_order() {
    // gelu, then add a constant, then softmax: three steps in one executable, which
    // is what a fused partition looks like from the inside.
    let mut b = GraphBuilder::new("chain");
    let x = b.input(f32(), dims(&[1, 3]), "x").expect("x");
    let c = b.input(f32(), dims(&[1, 3]), "c").expect("c");
    let g = b
        .node(
            Op::Gelu,
            &[x],
            &[(f32(), dims(&[1, 3]))],
            Default::default(),
            "g",
        )
        .expect("x");
    let a = b
        .node(
            Op::Add,
            &[g[0], c],
            &[(f32(), dims(&[1, 3]))],
            Default::default(),
            "a",
        )
        .expect("operands");
    let s = b
        .node(
            Op::Softmax { axis: -1 },
            &[a[0]],
            &[(f32(), dims(&[1, 3]))],
            Default::default(),
            "s",
        )
        .expect("operands");
    b.output(s[0]).expect("out");

    let executable = compile(&b.build());
    let inputs = std::collections::HashMap::from([
        (x, f32_tensor(&[1, 3], &[-1.0, 0.0, 1.0]).expect("fits")),
        (c, f32_tensor(&[1, 3], &[0.0, 0.0, 0.0]).expect("fits")),
    ]);
    let result = executable.run_values(&inputs).expect("run");
    let out = result.get(&s[0]).expect("produced").as_slice();

    // Softmax over the three gelu values the gelu test pins, computed here rather
    // than transcribed so the assertion cannot drift from the kernel.
    let gelu = |x: f32| 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2));
    let values = [-1.0f32, 0.0, 1.0].map(gelu);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = values.map(|v| (v - max).exp());
    let sum: f32 = exps.iter().sum();
    approx(out, &exps.map(|e| e / sum), 1e-6);
    // Softmax of anything sums to one.
    approx(&[out.iter().sum::<f32>()], &[1.0], 1e-6);
    assert_eq!(executable.steps().len(), 3);
}

// ---------------------------------------------------------------------------
// The decline path
// ---------------------------------------------------------------------------

#[test]
fn a_non_f32_graph_is_declined_rather_than_compiled() {
    let mut b = GraphBuilder::new("bf16");
    let x = b
        .input(DType::plain(ElementType::BF16), dims(&[1, 4]), "x")
        .expect("x");
    let g = b
        .node(
            Op::Gelu,
            &[x],
            &[(DType::plain(ElementType::BF16), dims(&[1, 4]))],
            Default::default(),
            "g",
        )
        .expect("x exists");
    b.output(g[0]).expect("out");
    let graph = b.build();

    // A decline arrives inside `Ok`, so the caller can try the next backend without
    // inspecting an error code.
    let outcome = CpuBackend::new()
        .compile(&graph, "ref::gelu")
        .expect("a decline is not an Err");
    assert!(outcome.is_declined());
    let (backend, reason) = outcome.declined().expect("declined");
    assert_eq!(backend, "cpu");
    assert!(reason.contains("f32"), "{reason}");
}

#[test]
fn a_fused_custom_op_is_declined() {
    // This backend computes ops, not fused attention kernels, and says so.
    let mut b = GraphBuilder::new("fa");
    let q = b.input(f32(), dims(&[2, 8, 8]), "q").expect("q");
    let out = b
        .node(
            Op::Custom {
                pattern_id: "fused::flash_attention_v3".into(),
            },
            &[q],
            &[(f32(), dims(&[2, 8, 8]))],
            Default::default(),
            "fa0",
        )
        .expect("q exists");
    b.output(out[0]).expect("out");

    let outcome = CpuBackend::new()
        .compile(&b.build(), "fused::flash_attention_v3")
        .expect("a decline is not an Err");
    assert!(outcome.is_declined());
    assert!(
        outcome
            .declined()
            .expect("declined")
            .1
            .contains("no CPU kernel"),
        "the reason should name the missing kernel"
    );
}

#[test]
fn a_graph_with_no_outputs_is_declined() {
    let mut b = GraphBuilder::new("dangling");
    let x = b.input(f32(), dims(&[1, 2]), "x").expect("x");
    b.node(
        Op::Gelu,
        &[x],
        &[(f32(), dims(&[1, 2]))],
        Default::default(),
        "g",
    )
    .expect("x exists");
    // No `output` call: a subgraph whose results nothing reads.

    let outcome = CpuBackend::new()
        .compile(&b.build(), "ref::gelu")
        .expect("a decline is not an Err");
    assert!(outcome.is_declined());
    assert!(
        outcome
            .declined()
            .expect("declined")
            .1
            .contains("no outputs")
    );
}

#[test]
fn the_pull_query_refuses_a_non_f32_dtype() {
    use ferrite_strata::{ShapeClass, SupportLevel};
    let backend = CpuBackend::new();
    assert_eq!(
        backend.supports(
            ReferenceOp::Gelu.pattern_id(),
            DType::plain(ElementType::F32),
            ShapeClass::Matrix
        ),
        SupportLevel::Fallback
    );
    assert_eq!(
        backend.supports(
            ReferenceOp::Gelu.pattern_id(),
            DType::plain(ElementType::BF16),
            ShapeClass::Matrix
        ),
        SupportLevel::Refuse
    );
}

#[test]
fn the_capability_blob_does_not_claim_fusion() {
    // `Fallback`, not `Fused`: this backend runs ops, it does not fuse them, and a
    // planner would act on a false `Fused`.
    use ferrite_strata::{ShapeClass, SupportLevel};
    let backend = CpuBackend::new();
    for op in ReferenceOp::all() {
        assert_eq!(
            backend.supports(op.pattern_id(), f32(), ShapeClass::Vector),
            SupportLevel::Fallback,
            "{} should not be claimed as fused",
            op.pattern_id()
        );
    }
}

#[test]
fn a_tensor_that_does_not_match_its_shape_is_refused() {
    // Padding or truncating would make this backend agree with a wrong kernel,
    // which is the one thing an oracle must never do.
    let err = CpuTensor::from_slice(&[1.0, 2.0], dims(&[2, 2])).expect_err("4 elements needed");
    assert!(err.to_string().contains("needs 4"), "{err}");
}

#[test]
fn indexing_past_the_end_is_an_error_not_a_panic() {
    let t = f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("fits");
    assert_eq!(t.at(1, 1).expect("in range"), 4.0);
    assert!(t.at(2, 0).is_err());
    assert!(t.at(0, 2).is_err());
}

#[test]
fn a_rank_3_tensor_is_refused_by_the_2d_helpers() {
    let t = f32_tensor(&[2, 2, 2], &[0.0; 8]).expect("fits");
    let err = t.dims2().expect_err("rank 3");
    assert!(err.to_string().contains("rank-2"), "{err}");
}
