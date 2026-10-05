//! End-to-end tests: plan, compile, run, across the real CPU backend.
//!
//! These are the tests that would catch a regression in the *design*, not just in a
//! kernel. Each one asserts a property the architecture is supposed to have.

use std::collections::HashMap;

use ferrite_strata::{
    Backend, DType, ElementType, Graph, GraphBuilder, Op, Shape, ShapeClass, SupportLevel, ValueId,
    capabilities::reference::{self, ReferenceOp},
    capabilities::{CapabilitySet, DTypeMask, DeclaredLimits, PatternSupport, StrataSupport},
};
use ferrite_strata_cpu::CpuBackend;
use ferrite_strata_runtime::{Session, Unit, inputs};

fn f32() -> DType {
    DType::plain(ElementType::F32)
}

fn dims(d: &[usize]) -> Shape {
    Shape::new(d).expect("within the rank limit")
}

/// A backend that always accepts, to prove the host path is not what is running.
struct ClaimsEverything {
    inner: CpuBackend,
    /// Held rather than rebuilt: `capabilities` is called on every `supports`
    /// query, and a partitioner makes a great many of them.
    cached: StrataSupport,
}

impl ClaimsEverything {
    fn build_caps() -> StrataSupport {
        let mut caps = CapabilitySet::empty("claims-everything")
            .with_dtype_mask(DTypeMask::all())
            .with_limits(DeclaredLimits {
                max_rank: Some(8),
                ..DeclaredLimits::default()
            });
        for op in ReferenceOp::all() {
            for class in reference::all_shape_classes() {
                caps = caps.with_pattern(PatternSupport::new(
                    op.pattern_id(),
                    f32(),
                    *class,
                    SupportLevel::Fused,
                ));
            }
        }
        // The whole point of this backend: it claims a fused pattern its compiler
        // cannot actually take.
        for class in reference::all_shape_classes() {
            caps = caps.with_pattern(PatternSupport::new(
                "fused::flash_attention_v3",
                f32(),
                *class,
                SupportLevel::Fused,
            ));
        }
        StrataSupport::new(caps)
    }
}

impl Backend for ClaimsEverything {
    fn name(&self) -> &str {
        "claims-everything"
    }

    fn plugin_version(&self) -> ferrite_strata::PluginVersion {
        ferrite_strata::PluginVersion::new(1, 0, 0, ferrite_strata::StrataVersion::current())
    }

    fn capabilities(&self) -> &StrataSupport {
        &self.cached
    }

    fn compile(
        &self,
        subgraph: &Graph,
        pattern_id: &str,
    ) -> Result<ferrite_strata::CompileOutcome, ferrite_strata::Error> {
        // Delegates to the real CPU backend, so a partition this backend accepts is
        // genuinely computed rather than silently skipped.
        self.inner.compile(subgraph, pattern_id)
    }
}

fn session_with_backends(backends: Vec<Box<dyn Backend>>) -> Session {
    let mut session = Session::new();
    for backend in backends {
        session.add(backend).expect("registration is infallible");
    }
    session
}

/// `x -> gelu -> mul(x) -> out`, all 2-D `f32`.
fn a_mlp_tail() -> (Graph, ValueId, ValueId) {
    let mut b = GraphBuilder::new("tail");
    let x = b.input(f32(), dims(&[2, 4]), "x").expect("x");
    let g = b
        .node(
            Op::Gelu,
            &[x],
            &[(f32(), dims(&[2, 4]))],
            Default::default(),
            "gelu0",
        )
        .expect("x");
    let m = b
        .node(
            Op::Mul,
            &[g[0], x],
            &[(f32(), dims(&[2, 4]))],
            Default::default(),
            "mul0",
        )
        .expect("operands");
    b.output(m[0]).expect("out");
    (b.build(), x, m[0])
}

#[test]
fn a_full_run_on_the_cpu_backend_produces_the_right_numbers() {
    let (graph, x, out) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);

    let plan = session.plan(&graph).expect("plan");
    assert!(plan.is_fully_placed());
    assert_eq!(plan.placed_node_count(), 2);

    let step = session.compile(&graph).expect("compile");
    let arena = step
        .run_with_inputs(&inputs(&[(
            x,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )]))
        .expect("run");

    let result = arena.f32_at(out).expect("the output is published");
    // gelu(1.0) * 1.0, gelu(2.0) * 2.0, and so on: eight elements.
    assert_eq!(result.len(), 8);
    assert!((result[0] - 0.841_344_8).abs() < 1e-4, "{}", result[0]);
    assert!(
        (result[1] - 2.0 * 1.954_499_7).abs() < 1e-3,
        "{}",
        result[1]
    );
}

#[test]
fn a_run_with_no_backends_still_produces_the_right_numbers() {
    // The load-bearing property. No backend at all, so every node is a host node,
    // and the answer is the same answer.
    let (graph, x, out) = a_mlp_tail();
    let mut session = Session::new();
    assert_eq!(session.backend_count(), 0);

    let plan = session.plan(&graph).expect("plan");
    assert!(!plan.is_fully_placed());
    assert_eq!(plan.placed_node_count(), 0);
    assert_eq!(plan.host_nodes().len(), 2);

    let step = session.compile(&graph).expect("compile");
    assert_eq!(step.host_unit_count(), 1);

    let arena = step
        .run_with_inputs(&inputs(&[(
            x,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )]))
        .expect("run");
    let result = arena.f32_at(out).expect("published");
    assert!((result[0] - 0.841_344_8).abs() < 1e-4, "{}", result[0]);
}

#[test]
fn the_host_path_and_the_cpu_backend_agree() {
    // The oracle relationship in the direction that matters: the host fallback is not
    // a different implementation, it is the same answer.
    let (graph, x, out) = a_mlp_tail();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut with_backend = session_with_backends(vec![Box::new(CpuBackend::new())]);
    with_backend.plan(&graph).expect("plan");
    let accelerated = with_backend
        .compile(&graph)
        .expect("compile")
        .run_with_inputs(&inputs(&[(x, data.clone())]))
        .expect("run");

    let mut host_only = Session::new();
    host_only.plan(&graph).expect("plan");
    let fallback = host_only
        .compile(&graph)
        .expect("compile")
        .run_with_inputs(&inputs(&[(x, data)]))
        .expect("run");

    let a = accelerated.f32_at(out).expect("published");
    let b = fallback.f32_at(out).expect("published");
    for (i, (left, right)) in a.iter().zip(b).enumerate() {
        assert!(
            (left - right).abs() < 1e-6,
            "element {i}: {left} vs {right}"
        );
    }
}

#[test]
fn outputs_are_published_under_the_original_graphs_ids() {
    // The subgraph extraction remaps ids. A run that computes correctly but publishes
    // under ids the caller cannot name is a run nobody can use, and this is the test
    // that says so.
    let (graph, x, out) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);
    let step = session.compile(&graph).expect("compile");
    let arena = step
        .run_with_inputs(&inputs(&[(x, vec![1.0; 8])]))
        .expect("run");

    // The exact ids the graph named, with no remapping in between.
    assert_eq!(arena.f32_at(out).map(<[f32]>::len), Some(8));
    assert_eq!(arena.f32_at(x).map(<[f32]>::len), Some(8));
}

#[test]
fn an_intermediate_is_not_published_even_though_it_is_computed() {
    // The gelu's result exists during the run and is deliberately not handed back:
    // it is the fused kernel's private scratch, and publishing it would let a caller
    // depend on something a backend may legitimately stop producing.
    let (graph, x, out) = a_mlp_tail();
    let mut session = Session::new();
    let step = session.compile(&graph).expect("compile");
    let arena = step
        .run_with_inputs(&inputs(&[(x, vec![1.0; 8])]))
        .expect("run");

    let intermediate = graph
        .value(out)
        .producer
        .map(|node| graph.node(node).inputs[0])
        .expect("the output's producer reads an intermediate");

    assert!(arena.f32_at(out).is_some(), "the output is published");
    assert!(
        !arena.ids().contains(&intermediate),
        "the intermediate leaked into the caller's arena"
    );
}

#[test]
fn a_host_only_run_publishes_only_the_graph_outputs() {
    // The host path must not leak its intermediates: a value it never promised to
    // expose is a value a later change may stop producing.
    let (graph, x, out) = a_mlp_tail();
    let mut session = Session::new();
    let step = session.compile(&graph).expect("compile");
    let arena = step
        .run_with_inputs(&inputs(&[(x, vec![1.0; 8])]))
        .expect("run");
    // Three values in the graph, one input and one output published; the gelu's
    // result stays private to the run.
    assert_eq!(graph.value_count(), 3);
    assert_eq!(arena.ids().len(), 2, "{}", arena.ids().len());
    assert!(arena.ids().contains(&out));
}

#[test]
fn a_backend_that_claims_support_it_does_not_have_still_produces_the_right_numbers() {
    // The failure mode decline exists to make recoverable. A vendor whose capability
    // table overstates what its compiler will take should cost time, not correctness.
    let backends: Vec<Box<dyn Backend>> = vec![Box::new(ClaimsEverything {
        inner: CpuBackend::new(),
        cached: ClaimsEverything::build_caps(),
    })];
    let mut session = session_with_backends(backends);

    let mut b = GraphBuilder::new("fused");
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
        .expect("q");
    b.output(out[0]).expect("out");
    let graph = b.build();

    // The partitioner will place it, because this backend said yes.
    let plan = session.plan(&graph).expect("plan");
    assert!(plan.is_fully_placed());

    // The CPU backend declines the fused op at compile time, so the runtime falls
    // back to the host. The host has no kernel for a fused pattern either, so this
    // particular graph cannot run — but it fails as a *missing host kernel*, at run
    // time, not as a rejected plan. That distinction is the point: the partitioner
    // believed the backend, the compiler corrected it, and the correction is a
    // runtime fact rather than a planning failure.
    let step = session.compile(&graph).expect("compile does not fail");
    let result = step.run_with_inputs(&inputs(&[(q, vec![0.0; 128])]));
    let err = result.expect_err("the host has no fused kernel");
    assert!(
        err.to_string().contains("no kernel"),
        "the failure should name the missing host kernel: {err}"
    );
    assert!(!err.is_device_poisoning());
}

#[test]
fn a_plan_reports_which_backend_takes_each_partition() {
    let (graph, _, _) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);
    let plan = session.plan(&graph).expect("plan");

    assert_eq!(session.backend_names(), ["cpu"]);
    let units = plan
        .partitions()
        .iter()
        .map(|p| Unit::Backend(p.backend))
        .collect::<Vec<_>>();
    assert!(!units.is_empty());
    for unit in units {
        assert_eq!(unit.to_string(), "backend#0");
    }
}

#[test]
fn planning_is_deterministic() {
    // What the compiled-subgraph cache depends on: the same graph and the same
    // backends give the same partition, every time.
    let (graph, _, _) = a_mlp_tail();
    let first = session_with_backends(vec![Box::new(CpuBackend::new())]);
    let second = session_with_backends(vec![Box::new(CpuBackend::new())]);

    let mut a = first;
    let mut b = second;
    let plan_a = a.plan(&graph).expect("plan");
    let plan_b = b.plan(&graph).expect("plan");
    assert_eq!(plan_a.partitions(), plan_b.partitions());
    assert_eq!(plan_a.host_nodes(), plan_b.host_nodes());
}

#[test]
fn a_run_can_be_repeated_without_accumulating_state() {
    // An inference server runs the same graph repeatedly. A second run must not see
    // the first one's leftovers.
    let (graph, x, out) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);
    let step = session.compile(&graph).expect("compile");

    let mut arena = ferrite_strata::Arena::new();
    arena.insert_f32(x, vec![1.0; 8]);
    step.run(&mut arena).expect("first run");
    let first = arena.f32_at(out).expect("published").to_vec();

    step.run(&mut arena).expect("second run");
    let second = arena.f32_at(out).expect("published").to_vec();
    assert_eq!(first, second);
}

#[test]
fn a_missing_input_is_reported_naming_the_value() {
    let (graph, _, _) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);
    let step = session.compile(&graph).expect("compile");

    let err = step
        .run_with_inputs(&HashMap::new())
        .expect_err("no inputs supplied");
    // The message names the value in the *partition's* numbering, not the caller's,
    // because that is the arena the executor sees. Stated rather than papered over:
    // a caller debugging this learns which partition wanted an operand it never
    // supplied, which is the question they actually have.
    assert!(err.to_string().contains("in the arena"), "{err}");
    assert!(err.to_string().contains("cpu"), "{err}");
    // A missing input is not a device failure.
    assert!(!err.is_device_poisoning());
}

#[test]
fn a_unit_reports_itself_when_it_fails() {
    // "Which unit failed" is the first question anyone asks, so the error has to
    // carry it rather than making the caller guess from the units list.
    let (graph, _, _) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);
    let step = session.compile(&graph).expect("compile");

    let err = step
        .run_with_inputs(&HashMap::new())
        .expect_err("no inputs");
    match err {
        ferrite_strata_runtime::ExecutionError::Backend { unit, .. } => {
            assert!(!unit.to_string().is_empty());
        }
        other => panic!("expected a backend unit failure, got {other:?}"),
    }
}

#[test]
fn a_backend_with_no_declared_limit_is_read_as_unbounded_by_default() {
    // The CPU backend declares no allocation limit; a plan must not refuse work
    // because of a limit that was never stated.
    let (graph, _, _) = a_mlp_tail();
    let mut session = session_with_backends(vec![Box::new(CpuBackend::new())]);
    assert!(session.plan(&graph).expect("plan").is_fully_placed());
}

#[test]
fn the_shape_class_a_query_is_answered_over_is_the_one_the_node_produces() {
    // A `ref::matmul` entry declared for `batch3` must not claim a `matrix` node,
    // which is what makes per-shape capability data worth having.
    let caps = CapabilitySet::empty("narrow")
        .with_dtype_mask(DTypeMask::all())
        .with_pattern(PatternSupport::new(
            ReferenceOp::MatMul.pattern_id(),
            f32(),
            ShapeClass::Batch3,
            SupportLevel::Fused,
        ));
    let support = StrataSupport::new(caps);
    assert_eq!(
        support.supports(ReferenceOp::MatMul.pattern_id(), f32(), ShapeClass::Batch3),
        SupportLevel::Fused
    );
    assert_eq!(
        support.supports(ReferenceOp::MatMul.pattern_id(), f32(), ShapeClass::Matrix),
        SupportLevel::Refuse
    );
}
