//! The compact form (`compact`) against the plain one.

use super::kernel::{compact_mode, translate_full, translate_full_as, KernelOptions};
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::runner::{self, KernelState};
use crate::gpu::recomp::KERNEL_LANES;

fn repeated(copies: usize) -> Vec<u8> {
    let i = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut v = Vec::new();
    for _ in 0..copies {
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)]));
        v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 256), ("VSRC1", 1)]));
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 130)]));
        v.extend(i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", 5), ("VSRC1", 0)]));
        v.extend(i("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)]));
        v.extend(i("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]));
        v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 256), ("VSRC1", 3)]));
        v.extend(i("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 6)]));
        v.extend(i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 2)]));
        v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]));
        v.extend(i("S_ADD_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 3), ("SSRC1", 129)]));
    }
    v.extend(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 133)]));
    v.extend(i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 256), ("VSRC1", 2)]));
    v.extend(i("S_SUB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)]));
    v.extend(i("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]));
    v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)]));
    v.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
    v
}

fn size(copies: usize, o: KernelOptions, compact: bool) -> usize {
    translate_full_as(&repeated(copies), o, compact).unwrap_or_else(|e| panic!("{copies} copies, compact {compact}: {e}")).dxil.len()
}

#[test]
fn the_compact_form_lowers_a_repeated_block_once() {
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let (compact, plain) = (size(128, o, true) - size(64, o, true), size(128, o, false) - size(64, o, false));
        assert!(compact * 20 < plain, "wave64 {wave64} split {split}: 64 more copies cost {compact} B compact, {plain} B plain");
    }
}

#[test]
fn a_large_program_that_repeats_itself_takes_the_compact_form() {
    if compact_mode().is_some() {
        return;
    }
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    for (copies, compact) in [(2500, true), (2000, false)] {
        assert_eq!(translate_full(&repeated(copies), o).unwrap().dxil.len(), size(copies, o, compact), "{copies} copies");
    }
}

#[test]
fn gpu_compact_form_runs_as_the_plain_form() {
    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, _, d) => return runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
    }
    let mut st = KernelState::default();
    (st.sgpr[1], st.sgpr[2]) = (3, 12);
    for lane in 0..KERNEL_LANES {
        st.vgpr[lane][0] = lane as u32;
    }
    let code = repeated(40);
    let o = KernelOptions { wave64: true, split: runner::split_wave64(), mem_windows: 1, sampler_base: 0 };
    let run = |compact: bool| {
        let k = translate_full_as(&code, o, compact).unwrap_or_else(|e| panic!("compact {compact}: {e}"));
        runner::run_with_statics(&k.dxil, &st, &[], &[], &[], &k.static_samplers).expect("GPU run")
    };
    let (c, p) = (run(true), run(false));
    assert_eq!(&c.sgpr[..6], &[120, 3, 12, 1, 0, 80]);
    for lane in 0..KERNEL_LANES as u32 {
        let masked = lane * (lane.saturating_sub(1) / 2);
        assert_eq!(&c.vgpr[lane as usize][..4], &[lane, 40 * lane, 5 * lane, masked], "lane {lane}");
    }
    assert_eq!((c.sgpr, c.vgpr, c.scc), (p.sgpr, p.vgpr, p.scc));
}

#[test]
fn gpu_compact_form_spends_the_loop_budget_as_the_plain_form() {
    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, _, d) => return runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
    }
    let mut v = Vec::new();
    v.extend(encode("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None).unwrap());
    v.extend(encode("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 255)], Some(super::lower::LOOP_BUDGET + 1)).unwrap());
    v.extend(encode("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None).unwrap());
    v.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 129)], None).unwrap());
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let o = KernelOptions { wave64: true, split: runner::split_wave64(), mem_windows: 1, sampler_base: 0 };
    let run = |compact: bool| {
        let k = translate_full_as(&v, o, compact).unwrap_or_else(|e| panic!("compact {compact}: {e}"));
        runner::run_with_statics(&k.dxil, &KernelState::default(), &[], &[], &[], &k.static_samplers).expect("GPU run")
    };
    let (c, p) = (run(true), run(false));
    assert_eq!(&c.sgpr[..2], &[super::lower::LOOP_BUDGET, 0]);
    assert_eq!(&c.sgpr[..2], &p.sgpr[..2]);
}

#[test]
fn a_mask_read_whole_after_one_copy_only_splits_the_class() {
    let i = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let cmp = |v: &mut Vec<u8>| {
        v.extend(i("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)]));
        v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    };
    let mut v = Vec::new();
    cmp(&mut v);
    v.extend(i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128), ("VSRC1", 3)]));
    v.extend(i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    cmp(&mut v);
    v.extend(i("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]));
    v.extend(i("S_ENDPGM", "ENC_SOPP", &[]));
    let cfg = super::cfg::Cfg::build(&v).unwrap();
    let leaders = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    let lw = super::lower::Lw::new(&cfg.insts, &leaders, o, &super::kernel::Entry::Kernel);
    let p = super::compact::plan(&cfg, &lw).unwrap();
    assert_eq!(p.order.len(), 4);
    assert_ne!(p.class[0], p.class[2]);
}

fn distinct_classes(classes: usize, with_loop: bool) -> Vec<u8> {
    let jump_blocks = classes - if with_loop { 2 } else { 1 };
    let mut v = Vec::new();
    for k in 0..jump_blocks {
        v.extend(encode("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 255)], Some(k as u32 + 1)).unwrap());
        v.extend(encode("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap());
    }
    if with_loop {
        v.extend(encode("S_SUB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)], None).unwrap());
        v.extend(encode("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)], None).unwrap());
        v.extend(encode("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None).unwrap());
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

struct DisasmCfg {
    switches: usize,
    max_switch_targets: usize,
    max_fan_in: usize,
    latch_fan_in: Option<usize>,
}

fn disasm_cfg(dis: &str, block_count: usize) -> DisasmCfg {
    let mut in_fn = false;
    let mut cur_label: Option<String> = None;
    let mut latch_label: Option<String> = None;
    let mut switches = 0usize;
    let mut in_switch = false;
    let mut switch_targets = 0usize;
    let mut max_switch_targets = 0usize;
    let mut in_degree: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let latch_suffix = format!(", {block_count}");
    for line in dis.lines() {
        let t = line.trim();
        if t.starts_with("define ") {
            in_fn = true;
            cur_label = None;
            continue;
        }
        if !in_fn {
            continue;
        }
        if t == "}" {
            break;
        }
        if let Some(rest) = t.strip_prefix("; <label>:") {
            cur_label = rest.trim_start().split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.').next().filter(|s| !s.is_empty()).map(str::to_string);
            continue;
        }
        let code = t.split(';').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        if cur_label.is_some() && code.contains("icmp uge i32 ") && code.ends_with(&latch_suffix) {
            latch_label = cur_label.clone();
        }
        if code.starts_with("switch i32 ") {
            switches += 1;
            in_switch = true;
            switch_targets = 0;
        }
        let mut rest = code;
        while let Some(pos) = rest.find("label %") {
            rest = &rest[pos + "label %".len()..];
            let end = rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.').unwrap_or(rest.len());
            let target = &rest[..end];
            if !target.is_empty() {
                *in_degree.entry(target.to_string()).or_insert(0) += 1;
                if in_switch {
                    switch_targets += 1;
                }
            }
            rest = &rest[end..];
        }
        if in_switch && code.ends_with(']') {
            max_switch_targets = max_switch_targets.max(switch_targets);
            in_switch = false;
        }
    }
    DisasmCfg {
        switches,
        max_switch_targets,
        max_fan_in: in_degree.values().copied().max().unwrap_or(0),
        latch_fan_in: latch_label.and_then(|l| in_degree.get(&l).copied()),
    }
}

#[test]
fn large_class_counts_use_nested_switch_dispatch_with_bounded_fan_in() {
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    for (classes, with_loop) in [(65, false), (65, true), (100, false)] {
        let code = distinct_classes(classes, with_loop);
        let cfg = super::cfg::Cfg::build(&code).unwrap();
        let leaders = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
        let lw = super::lower::Lw::new(&cfg.insts, &leaders, o, &super::kernel::Entry::Kernel);
        let p = super::compact::plan(&cfg, &lw).unwrap();
        assert_eq!(p.rep.len(), classes, "classes={classes} with_loop={with_loop}");
        assert_eq!(!cfg.headers.is_empty(), with_loop);

        let k = translate_full_as(&code, o, true).unwrap_or_else(|e| panic!("classes={classes} with_loop={with_loop}: {e}"));
        super::validate::validate(&k.dxil).unwrap_or_else(|e| panic!("validate classes={classes} with_loop={with_loop}: {e}"));
        let dis = super::validate::disassemble(&k.dxil).expect("disassemble");
        let stats = disasm_cfg(&dis, p.order.len());
        let groups = classes.div_ceil(32);
        let expected_switches = 1 + classes / 32 + usize::from(classes % 32 > 1);
        assert_eq!(stats.switches, expected_switches, "classes={classes} with_loop={with_loop}: expected outer + inner switches");
        assert!(stats.max_switch_targets <= 32, "classes={classes}: switch targets {}", stats.max_switch_targets);
        assert!(stats.max_fan_in <= 32, "classes={classes} with_loop={with_loop}: max fan-in {}", stats.max_fan_in);
        let latch_fan_in = stats.latch_fan_in.expect("latch block in disassembly");
        assert!(latch_fan_in <= groups + 2, "classes={classes}: latch fan-in {latch_fan_in} > {}", groups + 2);
        assert_eq!(latch_fan_in, groups, "classes={classes} with_loop={with_loop}: exact latch fan-in");
    }
}

#[test]
fn small_class_counts_keep_single_flat_switch() {
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    for classes in [16, 32] {
        let code = distinct_classes(classes, false);
        let cfg = super::cfg::Cfg::build(&code).unwrap();
        let leaders = cfg.blocks.iter().filter_map(|b| b.insts.first().map(|&i| cfg.insts[i].pc)).collect();
        let lw = super::lower::Lw::new(&cfg.insts, &leaders, o, &super::kernel::Entry::Kernel);
        let p = super::compact::plan(&cfg, &lw).unwrap();
        assert_eq!(p.rep.len(), classes);

        let k = translate_full_as(&code, o, true).unwrap_or_else(|e| panic!("classes={classes}: {e}"));
        super::validate::validate(&k.dxil).unwrap_or_else(|e| panic!("validate classes={classes}: {e}"));
        let dis = super::validate::disassemble(&k.dxil).expect("disassemble");
        let stats = disasm_cfg(&dis, p.order.len());
        assert_eq!(stats.switches, 1, "classes={classes}: expected single flat switch");
        assert_eq!(stats.max_switch_targets, classes);
    }
}
