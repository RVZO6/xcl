//! Memory window selection and the windows taken from the descriptor heap: the direct
//! translator's form of the HLSL kernel's `window_read` / `window_write` and of its
//! `xcl_m*` accessors with `Options::heap_windows` (`recomp::emit::mem_accessors`).

use super::lower::{Lw, VAR_WIN};
use super::module::{Pred, Value};
use crate::gpu::rdna2::Decoded;

/// A term of a 64-bit effective address (`window_at`).
#[derive(Clone, Copy)]
pub enum Ea {
    /// a 32-bit value, zero-extended
    U(Value),
    /// a 32-bit constant, zero-extended
    K(u32),
    /// a 32-bit value, sign-extended
    S(Value),
    /// the 64-bit product of two 32-bit values
    Mul(Value, Value),
}

impl Lw {
    /// The program may address several memory windows.
    pub fn windowed(&self) -> bool {
        self.mems.len() > 1
    }

    /// The high dword of the UMA base (`IO_UMA_BASE_HI` of the io buffer, `xcl_uma_base`),
    /// hoisted to the entry block so multi-window shaders load it once rather than on every
    /// memory operation.
    fn uma_base(&mut self) -> Value {
        if let Some(v) = self.uma_base_val {
            return v;
        }
        let io = self.init.io.expect("io");
        let off = self.c(crate::gpu::recomp::IO_UMA_BASE_HI as u32 * 4);
        let cur = self.cur();
        let entry = self.init.entry;
        let v = if cur == entry {
            self.s.raw_load(io, off, 1)[0]
        } else {
            let prev = self.s.m.set_insert_before_term(true);
            self.s.m.set_block(entry);
            let v = self.s.raw_load(io, off, 1)[0];
            self.s.m.set_block(cur);
            self.s.m.set_insert_before_term(prev);
            v
        };
        self.uma_base_val = Some(v);
        v
    }

    /// `i1` → `i32` 0 / 1, folded when constant.
    fn zext_i1(&mut self, c: Value) -> Value {
        match self.s.m.const_u64(c) {
            Some(v) => self.c(v as u32),
            None => self.u(c),
        }
    }

    /// 64-bit `(hi, lo) + u32` in pure 32-bit carry arithmetic.
    fn add64_u32(&mut self, hi: Value, lo: Value, d_lo: Value) -> (Value, Value) {
        if self.s.m.const_u64(d_lo) == Some(0) {
            return (hi, lo);
        }
        if self.s.m.const_u64(lo) == Some(0) {
            return (hi, d_lo);
        }
        let new_lo = self.add(lo, d_lo);
        let carry = self.cmp(Pred::ULt, new_lo, lo);
        let carry = self.zext_i1(carry);
        (self.add(hi, carry), new_lo)
    }

    /// 64-bit `(a_hi, a_lo) + (b_hi, b_lo)` in pure 32-bit carry arithmetic.
    fn add64_u64(&mut self, a_hi: Value, a_lo: Value, b_hi: Value, b_lo: Value) -> (Value, Value) {
        let hi = self.add(a_hi, b_hi);
        self.add64_u32(hi, a_lo, b_lo)
    }

    /// 64-bit `(a_hi, a_lo) - (b_hi, b_lo)` in pure 32-bit borrow arithmetic.
    fn sub64_u64(&mut self, a_hi: Value, a_lo: Value, b_hi: Value, b_lo: Value) -> (Value, Value) {
        let hi = self.sub(a_hi, b_hi);
        if self.s.m.const_u64(b_lo) == Some(0) {
            return (hi, a_lo);
        }
        if a_lo == b_lo {
            return (hi, self.c(0));
        }
        let new_lo = self.sub(a_lo, b_lo);
        let borrow = self.cmp(Pred::ULt, a_lo, b_lo);
        let borrow = self.zext_i1(borrow);
        (self.sub(hi, borrow), new_lo)
    }

    /// 64-bit `(hi, lo) + sext(d_i32)` in pure 32-bit carry / borrow arithmetic.
    fn add64_i32(&mut self, hi: Value, lo: Value, d_i32: Value) -> (Value, Value) {
        match self.s.m.const_u64(d_i32) {
            Some(0) => (hi, lo),
            Some(k) if (k as u32 as i32) >= 0 => self.add64_u32(hi, lo, d_i32),
            Some(k) => {
                let mag = self.c((k as u32).wrapping_neg());
                let zero = self.c(0);
                self.sub64_u64(hi, lo, zero, mag)
            }
            None => {
                let d_hi = self.ashrc(d_i32, 31);
                self.add64_u64(hi, lo, d_hi, d_i32)
            }
        }
    }

    /// 64-bit `(hi, lo) + u32 * u32` using DXIL 32-bit wide multiply (`umul_wide`).
    fn add64_mul(&mut self, hi: Value, lo: Value, a: Value, b: Value) -> (Value, Value) {
        match (self.s.m.const_u64(a), self.s.m.const_u64(b)) {
            (Some(0), _) | (_, Some(0)) => (hi, lo),
            (Some(1), _) => self.add64_u32(hi, lo, b),
            (_, Some(1)) => self.add64_u32(hi, lo, a),
            (Some(x), Some(y)) => {
                let p = x * y;
                let (p_hi, p_lo) = (self.c((p >> 32) as u32), self.c(p as u32));
                self.add64_u64(hi, lo, p_hi, p_lo)
            }
            _ => {
                let (p_hi, p_lo) = self.s.umul_wide(a, b);
                self.add64_u64(hi, lo, p_hi, p_lo)
            }
        }
    }

    /// Selects the window of an access whose 64-bit effective address is `hi:lo` plus
    /// `adds` (carry / borrow into the high dword), as the HLSL kernel: a read with
    /// `hi == 0` (32-bit console descriptor table pointers) goes to window 0; any other
    /// high dword outside `[uma_base, uma_base + windows)` selects no window (reads 0,
    /// stores and atomics are dropped). Several windows only.
    pub fn window_at(&mut self, hi: Value, lo: Value, adds: &[Ea], write: bool) {
        if !self.windowed() {
            return;
        }
        let (mut hi, mut lo) = (hi, lo);
        for &t in adds {
            (hi, lo) = match t {
                Ea::U(v) => self.add64_u32(hi, lo, v),
                Ea::K(0) => (hi, lo),
                Ea::K(k) => {
                    let k = self.c(k);
                    self.add64_u32(hi, lo, k)
                }
                Ea::S(v) => self.add64_i32(hi, lo, v),
                Ea::Mul(a, b) => self.add64_mul(hi, lo, a, b),
            };
        }
        self.window_hi(hi, write);
    }

    /// `window_at` of a high dword already computed (`window_read` / `window_write`).
    pub fn window_hi(&mut self, hi: Value, write: bool) {
        if !self.windowed() {
            return;
        }
        let base = self.uma_base();
        let mut w = self.sub(hi, base);
        if !write {
            let z = self.cmpc(Pred::Eq, hi, 0);
            let zero = self.c(0);
            w = self.sel(z, zero, w);
        }
        self.wr(VAR_WIN, w);
    }

    /// The window of a buffer access: V# base `d1[15:0]:d0` plus `adds` (the stride bits
    /// `d1[29:16]` are not part of the address).
    pub fn buffer_window(&mut self, d0: Value, d1: Value, adds: &[Ea], write: bool) {
        if self.windowed() {
            let hi = self.andc(d1, 0xFFFF);
            self.window_at(hi, d0, adds, write);
        }
    }

    /// The window of a FLAT / global access: SADDR pair + VGPR offset, or the 64-bit VGPR
    /// address, plus the instruction offset (signed for global).
    pub fn flat_window(&mut self, d: &Decoded, write: bool) {
        if !self.windowed() {
            return;
        }
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let saddr = Self::fv(d, "SADDR");
        let addr = Self::fv(d, "ADDR");
        let raw = Self::fv(d, "OFFSET") as u32;
        let off = if global {
            Ea::S(self.c((((raw << 20) as i32) >> 20) as u32))
        } else {
            Ea::U(self.c(raw))
        };
        if global && saddr != 0x7D && saddr != 0x7F {
            let (lo, hi, v) = (self.sreg(saddr), self.sreg(saddr + 1), self.vreg(addr));
            self.window_at(hi, lo, &[Ea::U(v), off], write);
        } else {
            let (lo, hi) = (self.vreg(addr), self.vreg(addr + 1));
            self.window_at(hi, lo, &[off], write);
        }
    }

    /// The window of a `global_*_addtid` access: SADDR pair (or 0) + 4 × lane + offset.
    pub fn addtid_window(&mut self, d: &Decoded, write: bool) {
        if !self.windowed() {
            return;
        }
        let global = d.enc.name == "ENC_FLAT_GLBL";
        let saddr = Self::fv(d, "SADDR");
        let raw = Self::fv(d, "OFFSET") as u32;
        let off = if global {
            Ea::S(self.c((((raw << 20) as i32) >> 20) as u32))
        } else {
            Ea::U(self.c(raw))
        };
        let lane = self.lane;
        let l4 = self.shlc(lane, 2);
        let (lo, hi) = match saddr {
            0x7D | 0x7F => (self.c(0), self.c(0)),
            _ => (self.sreg(saddr), self.sreg(saddr + 1)),
        };
        self.window_at(hi, lo, &[Ea::U(l4), off], write);
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::dxil::{kernel, validate};
    use crate::gpu::{rdna2, recomp};

    fn asm(list: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
        let mut code = Vec::new();
        for &(name, enc, fields) in list {
            code.extend(rdna2::encode(name, enc, fields, None).unwrap_or_else(|e| panic!("encode {name}: {e}")));
        }
        code.extend(rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        code
    }

    fn sfi0_flags(dxil: &[u8]) -> u64 {
        let n = u32::from_le_bytes(dxil[28..32].try_into().unwrap()) as usize;
        for i in 0..n {
            let off = u32::from_le_bytes(dxil[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
            if &dxil[off..off + 4] == b"SFI0" {
                return u64::from_le_bytes(dxil[off + 8..off + 16].try_into().unwrap());
            }
        }
        panic!("missing SFI0 part");
    }

    fn assert_no_int64(dxil: &[u8], label: &str) -> String {
        validate::validate(dxil).unwrap_or_else(|e| panic!("{label} validate: {e}"));
        assert_eq!(sfi0_flags(dxil) & (1 << 15), 0, "{label}: SFI0 bit 15 (Int64Ops) must be 0");
        let dis = validate::disassemble(dxil).unwrap_or_else(|e| panic!("{label} disassemble: {e}"));
        assert!(!dis.contains("64-bit integers"), "{label}: must not require Int64Ops:\n{dis}");
        let body = dis.split("define void @main()").nth(1).unwrap_or(&dis).split("\n}\n").next().unwrap_or(&dis);
        assert!(!body.contains(" i64 "), "{label}: must not emit i64 ops in @main:\n{body}");
        dis
    }

    #[test]
    fn multi_window_buffer_and_flat_drop_int64_ops() {
        let full = asm(&[
            ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 0), ("OFFSET", 0x10), ("SOFFSET", 2)]),
            ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 0), ("OFFSET", 0x1F_FFF0), ("SOFFSET", 125)]),
            ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 4), ("SOFFSET", 2)]),
            ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 2), ("OFFSET", 0x40), ("IDXEN", 1), ("OFFEN", 1)]),
            ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", 0x20), ("IDXEN", 1), ("OFFEN", 1)]),
            ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 5), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1), ("GLC", 1)]),
            ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 6), ("ADDR", 0), ("SADDR", 20), ("OFFSET", 0xFF0)]),
            ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 7), ("ADDR", 0), ("SADDR", 20), ("OFFSET", 0x800)]),
            ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 8), ("ADDR", 2), ("SADDR", 0x7D), ("OFFSET", 0xFFF)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 0), ("DATA", 6), ("SADDR", 20), ("OFFSET", 0x10)]),
            ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 9), ("ADDR", 0), ("DATA", 6), ("SADDR", 20), ("OFFSET", 0), ("GLC", 1)]),
            ("FLAT_LOAD_DWORD", "ENC_FLAT", &[("VDST", 10), ("ADDR", 2), ("OFFSET", 0x20)]),
            ("FLAT_STORE_DWORD", "ENC_FLAT", &[("ADDR", 2), ("DATA", 10), ("OFFSET", 0)]),
            ("GLOBAL_LOAD_DWORD_ADDTID", "ENC_FLAT_GLBL", &[("VDST", 11), ("SADDR", 20), ("OFFSET", 0xFFC)]),
            ("GLOBAL_LOAD_DWORD_ADDTID", "ENC_FLAT_GLBL", &[("VDST", 12), ("SADDR", 0x7D), ("OFFSET", 0)]),
            ("GLOBAL_STORE_DWORD_ADDTID", "ENC_FLAT_GLBL", &[("DATA", 11), ("SADDR", 20), ("OFFSET", 8)]),
        ]);
        for mem_windows in [4u32, 4 | recomp::HEAP_WINDOWS] {
            for wave64 in [true, false] {
                let opts = kernel::KernelOptions { wave64, split: false, mem_windows, sampler_base: 0 };
                let dxil = kernel::translate(&full, opts).expect("translate full");
                let dis = assert_no_int64(&dxil, &format!("full w={mem_windows:#x} w64={wave64}"));
                assert!(dis.contains("@dx.op.binaryWithTwoOuts.i32(i32 42,"), "expected 32-bit umul_wide:\n{dis}");
            }
        }
        let unindexed = asm(&[
            ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128)]),
            ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128)]),
            ("FLAT_LOAD_DWORD", "ENC_FLAT", &[("VDST", 5), ("ADDR", 2), ("OFFSET", 0)]),
        ]);
        let dxil = kernel::translate(&unindexed, kernel::KernelOptions { wave64: true, split: false, mem_windows: 4, sampler_base: 0 }).expect("translate unindexed");
        let dis = assert_no_int64(&dxil, "unindexed");
        assert!(!dis.contains("@dx.op.binaryWithTwoOuts.i32(i32 42,"), "zero-index Ea::Mul must fold away:\n{dis}");
    }

    #[test]
    fn window_carry_borrow_exact_boundary_cases() {
        let pack = |hi: u32, lo: u32| ((hi as u64) << 32) | (lo as u64);
        let unpack = |v: u64| ((v >> 32) as u32, v as u32);
        let add64_u32 = |hi: u32, lo: u32, d: u32| -> (u32, u32) {
            if d == 0 { return (hi, lo); }
            if lo == 0 { return (hi, d); }
            let nlo = lo.wrapping_add(d);
            (hi.wrapping_add((nlo < lo) as u32), nlo)
        };
        let add64_u64 = |ahi: u32, alo: u32, bhi: u32, blo: u32| add64_u32(ahi.wrapping_add(bhi), alo, blo);
        let sub64_u64 = |ahi: u32, alo: u32, bhi: u32, blo: u32| -> (u32, u32) {
            let hi = ahi.wrapping_sub(bhi);
            if blo == 0 { return (hi, alo); }
            (hi.wrapping_sub((alo < blo) as u32), alo.wrapping_sub(blo))
        };
        let add64_i32_const = |hi: u32, lo: u32, d: u32| match d {
            0 => (hi, lo),
            k if (k as i32) >= 0 => add64_u32(hi, lo, k),
            k => sub64_u64(hi, lo, 0, k.wrapping_neg()),
        };
        let add64_i32_dyn = |hi: u32, lo: u32, d: u32| add64_u64(hi, lo, ((d as i32) >> 31) as u32, d);
        let add64_mul = |hi: u32, lo: u32, a: u32, b: u32| {
            let p = (a as u64) * (b as u64);
            add64_u64(hi, lo, (p >> 32) as u32, p as u32)
        };
        let ult64 = |ahi: u32, alo: u32, bhi: u32, blo: u32| (ahi < bhi) || (ahi == bhi && alo < blo);
        let win = |hi: u32, base: u32, n: u32, write: bool| {
            let w = if !write && hi == 0 { 0 } else { hi.wrapping_sub(base) };
            (w < n).then_some(w)
        };

        let vals = [0u32, 1, 2, 16, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_F800, 0xFFFF_FFF0, 0xFFFF_FFFE, 0xFFFF_FFFF];
        for &hi in &[0u32, 1, 4, 7, 8, 0x7FFF_FFFF, 0xFFFF_FFFF] {
            for &lo in &vals {
                let base = pack(hi, lo);
                for &d in &vals {
                    assert_eq!(add64_u32(hi, lo, d), unpack(base.wrapping_add(d as u64)), "u32 ({hi:#x},{lo:#x})+{d:#x}");
                    let s_want = unpack(base.wrapping_add((d as i32 as i64) as u64));
                    assert_eq!(add64_i32_const(hi, lo, d), s_want, "i32c ({hi:#x},{lo:#x})+{d:#x}");
                    assert_eq!(add64_i32_dyn(hi, lo, d), s_want, "i32d ({hi:#x},{lo:#x})+{d:#x}");
                    for &bhi in &[0u32, 1, 0xFFFF_FFFF] {
                        let rhs = pack(bhi, d);
                        assert_eq!(add64_u64(hi, lo, bhi, d), unpack(base.wrapping_add(rhs)));
                        assert_eq!(sub64_u64(hi, lo, bhi, d), unpack(base.wrapping_sub(rhs)));
                        assert_eq!(ult64(hi, lo, bhi, d), base < rhs);
                    }
                }
            }
        }
        // Explicit boundary checks: lo=0+(-1), lo=MAX+1, i32::MIN, Ea::Mul overflow, va<base vs va>=base+size.
        assert_eq!(add64_i32_const(5, 0, (-1i32) as u32), (4, u32::MAX));
        assert_eq!(add64_i32_dyn(0, 0, (-1i32) as u32), (u32::MAX, u32::MAX));
        assert_eq!(add64_u32(4, u32::MAX, 1), (5, 0));
        assert_eq!(add64_u32(u32::MAX, u32::MAX, 1), (0, 0));
        for &(lo, want) in &[(0u32, (4, 0x8000_0000)), (0x7FFF_FFFF, (4, u32::MAX)), (0x8000_0000, (5, 0)), (u32::MAX, (5, 0x7FFF_FFFF))] {
            assert_eq!(add64_i32_const(5, lo, i32::MIN as u32), want);
            assert_eq!(add64_i32_dyn(5, lo, i32::MIN as u32), want);
        }
        assert_eq!(add64_mul(4, 0x8000_0000, 0x3FFF, 0x1000_0000), unpack(pack(4, 0x8000_0000) + 0x3FFFu64 * 0x1000_0000));
        assert_eq!(add64_mul(4, u32::MAX, u32::MAX, u32::MAX), unpack(pack(4, u32::MAX).wrapping_add(u32::MAX as u64 * u32::MAX as u64)));
        // Window bounds: base = 4<<32, size = 4<<32 (windows 0..4 => hi in 4..8).
        let (base_hi, n) = (4u32, 4u32);
        let (h_under, _) = add64_i32_const(base_hi, 0, (-1i32) as u32);
        assert!(ult64(h_under, u32::MAX, base_hi, 0) && win(h_under, base_hi, n, false).is_none() && win(h_under, base_hi, n, true).is_none());
        assert_eq!((win(0, base_hi, n, false), win(0, base_hi, n, true)), (Some(0), None));
        let (h_carry, _) = add64_u32(base_hi, 0xFFFF_FFF0, 0x20);
        assert_eq!(win(h_carry, base_hi, n, false), Some(1));
        let (h_cancel, _) = add64_i32_const(h_carry, 0x10, (-0x20i32) as u32);
        assert_eq!(win(h_cancel, base_hi, n, false), Some(0));
        assert_eq!(win(base_hi + n - 1, base_hi, n, true), Some(3));
        let (h_over, _) = add64_u32(base_hi + n - 1, u32::MAX, 1);
        assert!(!ult64(h_over, 0, base_hi + n, 0) && win(h_over, base_hi, n, false).is_none() && win(h_over, base_hi, n, true).is_none());
        let (h_back, _) = add64_i32_const(base_hi + n, 0, (-1i32) as u32);
        assert_eq!(win(h_back, base_hi, n, true), Some(3));
    }
}
