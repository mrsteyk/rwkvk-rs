use super::get_bf16;
use super::get_row_2d_bf16;

use std;

use half::prelude::*;
use safetensors::tensor::TensorView;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub(crate) struct StateElem {
    // 0
    pub ffn_x: Vec<f32>,
    // 1
    pub att_x: Vec<f32>,
    // 2
    pub att_a: Vec<f32>,
    // 3
    pub att_b: Vec<f32>,
    // 4
    pub att_p: Vec<f32>,
}

// SA means Sparse Attention?
#[derive(Debug)]
pub(crate) struct Att<'a> {
    // Can be 16
    pub key: TensorView<'a>,
    pub output: TensorView<'a>,
    pub receptance: TensorView<'a>,
    // ---
    // Can't be 16
    // pub time_decay: TensorView<'a>,
    // pub time_first: TensorView<'a>,
    pub time_decay: Vec<f32>,
    pub time_first: Vec<f32>,
    // Can be 16
    pub time_mix_k: TensorView<'a>,
    pub time_mix_r: TensorView<'a>,
    pub time_mix_v: TensorView<'a>,
    // ---
    pub value: TensorView<'a>,
}

impl Att<'_> {
    pub fn apply(&self, x: &[f32], state: &mut StateElem) -> Vec<f32> {
        // Most computations here are made&stored in f32 to save cycles.

        let xx = &state.att_x;

        let xop = |t| {
            std::iter::zip(
                x.iter()
                    .zip(get_bf16(t).iter())
                    .map(|(&x, &b)| x * b.to_f32()),
                get_bf16(t)
                    .iter()
                    .zip(xx.iter())
                    .map(|(&x, &xx)| xx * (1f32 - x.to_f32())),
            )
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>()
        };

        let xk = xop(&self.time_mix_k);
        let xv = xop(&self.time_mix_v);
        let xr = xop(&self.time_mix_r);
        state.att_x = x.to_vec();

        // mat by vec
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.receptance.shape()[1], xr.len());
        let rw = get_bf16(&self.receptance);
        // This improved tokenization by a lot lmfao
        let r = rw
            .chunks_exact(xr.len())
            .map(|x| {
                x.iter()
                    .zip(xr.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .map(|x| (1f32 / (1f32 + (-x).exp())))
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        debug_assert_eq!(self.key.shape()[1], xk.len());
        let kw = get_bf16(&self.key);
        let k = kw
            .chunks_exact(xk.len())
            .map(|x| {
                x.iter()
                    .zip(xk.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        debug_assert_eq!(self.value.shape()[1], xv.len());
        let vw = get_bf16(&self.value);
        let v = vw
            .chunks_exact(xv.len())
            .map(|x| {
                x.iter()
                    .zip(xv.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        let aa = &state.att_a;
        let bb = &state.att_b;
        let pp = &state.att_p;
        let ww = self
            .time_first
            .iter()
            .zip(k.iter())
            .map(|(&x, &y)| x + y)
            .collect::<Vec<_>>();
        let p = ww
            .iter()
            .zip(pp.iter())
            .map(|(&ww, &pp)| ww.max(pp))
            .collect::<Vec<_>>();
        let e1 = p
            .iter()
            .zip(pp.iter())
            .map(|(&p, &pp)| (pp - p).exp())
            .collect::<Vec<_>>();
        let e2 = p
            .iter()
            .zip(ww.iter())
            .map(|(&p, &ww)| (ww - p).exp())
            .collect::<Vec<_>>();

        let a = std::iter::zip(
            e1.iter().zip(aa.iter()).map(|(&e1, &aa)| e1 * aa),
            e2.iter().zip(v.iter()).map(|(&e2, &v)| e2 * v),
        )
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
        let b = e1
            .iter()
            .zip(bb.iter())
            .map(|(&e1, &bb)| e1 * bb)
            .zip(e2.iter())
            .map(|(x, &e2)| x + e2)
            .collect::<Vec<_>>();

        let ww = self
            .time_decay
            .iter()
            .zip(pp.iter())
            .map(|(&time_decay, &pp)| pp + time_decay)
            .collect::<Vec<_>>();
        let p = ww
            .iter()
            .zip(k.iter())
            .map(|(&ww, &k)| ww.max(k))
            .collect::<Vec<_>>();
        let e1 = p
            .iter()
            .zip(ww.iter())
            .map(|(&p, &ww)| (ww - p).exp())
            .collect::<Vec<_>>();
        let e2 = p
            .iter()
            .zip(k.iter())
            .map(|(&p, &k)| (k - p).exp())
            .collect::<Vec<_>>();

        // Almost there
        state.att_a = e1
            .iter()
            .zip(e2.iter())
            .zip(v.iter().zip(aa.iter()))
            .map(|((e1, e2), (&v, &aa))| e1 * aa + e2 * v)
            .collect();
        state.att_b = e1
            .iter()
            .zip(e2.iter())
            .zip(bb.iter())
            .map(|((e1, e2), &bb)| e1 * bb + e2)
            .collect();
        state.att_p = p;

        let wkv = a.iter().zip(b.iter()).map(|(&a, &b)| a / b);
        let rwkv = r
            .iter()
            .zip(wkv)
            .map(|(&r, wkv)| r * wkv)
            .collect::<Vec<_>>();

        let ow = get_bf16(&self.output);
        let ow_len = self.output.shape()[0];
        ow.chunks_exact(ow_len)
            .map(|x| {
                x.iter()
                    .zip(rwkv.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .collect()
    }
}

#[derive(Debug)]
pub(crate) struct Ffn<'a> {
    // Can be 16
    pub key: TensorView<'a>,
    pub receptance: TensorView<'a>,
    // ---
    pub time_mix_k: TensorView<'a>,
    pub time_mix_r: TensorView<'a>,
    // ---
    pub value: TensorView<'a>,
}

impl Ffn<'_> {
    pub fn apply(&self, x: &[f32], state: &mut StateElem) -> Vec<f32> {
        let xx = &state.ffn_x;
        let xk = std::iter::zip(
            x.iter()
                .zip(get_bf16(&self.time_mix_k).iter())
                .map(|(&x, &b)| x * b.to_f32()),
            get_bf16(&self.time_mix_k)
                .iter()
                .map(|&x| 1f32 - x.to_f32())
                .zip(xx.iter())
                .map(|(x, &y)| x * y),
        )
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
        let xr = std::iter::zip(
            x.iter()
                .zip(get_bf16(&self.time_mix_r).iter())
                .map(|(&x, &b)| x * b.to_f32()),
            get_bf16(&self.time_mix_r)
                .iter()
                .map(|&x| 1f32 - x.to_f32())
                .zip(xx.iter())
                .map(|(x, &y)| x * y),
        )
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
        state.ffn_x = x.to_vec();

        // mat by vec
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.receptance.shape()[1], xr.len());
        let rw = get_bf16(&self.receptance);
        let r = rw
            .chunks_exact(xr.len())
            .map(|x| {
                x.iter()
                    .zip(xr.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .map(|x| (1f32 / (1f32 + (-x).exp())))
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        debug_assert_eq!(self.key.shape()[1], xk.len());
        let kw = get_bf16(&self.key);
        // println!("{:?} {:?}", self.key.shape(), xk.len());
        let k = kw
            .chunks_exact(xk.len())
            .map(|x| {
                x.iter()
                    .zip(xk.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .map(|x| if x.is_sign_positive() { x } else { 0f32 })
            .map(|x| x * x)
            .collect::<Vec<_>>();

        let vw = get_bf16(&self.value);
        let kv = vw.chunks_exact(k.len()).map(|x| {
            x.iter()
                .zip(k.iter())
                .map(|(&x, &y)| x.to_f32() * y)
                .sum::<f32>()
        });

        r.iter().zip(kv).map(|(&x, y)| x * y).collect()
    }
}

#[derive(Debug)]
pub(crate) struct Ln<'a> {
    pub weight: TensorView<'a>,
    pub bias: TensorView<'a>,
}

impl Ln<'_> {
    pub fn apply_f32(&self, other: &[f32]) -> Option<Vec<f32>> {
        if self.weight.shape()[0] != other.len() {
            None
        } else {
            // TODO: is converting to f32 worth it?
            pub(crate) const NORM_EPS: f32 = 1e-5;

            // What the fuck
            // Collecting is a bit faster???
            let lenf = other.len() as f32;
            let mean: f32 = other.iter().sum();
            let mean = mean / lenf;
            let other = other.iter().map(|&x| x - mean).collect::<Vec<_>>();
            let sum2: f32 = other.iter().map(|x| x * x).sum();
            let sum2 = sum2 / (lenf) + NORM_EPS;
            // let sum = sum2.sqrt(); // this is slower?
            // this is same speed as mutating the array normally, rustc is so much smarter now lmao
            let other = other.iter().map(|x| (x / sum2.sqrt()));

            let w = get_bf16(&self.weight);
            let b = get_bf16(&self.bias);

            Some(
                other
                    .zip(w.iter())
                    .zip(b.iter())
                    .map(|((x, &w), &b)| (x * w.to_f32()) + b.to_f32())
                    .collect(),
            )
        }
    }

    pub fn apply_f32_f(other: &[f32], w: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        if w.len() != other.len() {
            None
        } else {
            // TODO: is converting to f32 worth it?
            pub(crate) const NORM_EPS: f32 = 1e-5;

            // What the fuck
            // Collecting is a bit faster???
            let lenf = other.len() as f32;
            let mean: f32 = other.iter().sum();
            let mean = mean / lenf;
            let other = other.iter().map(|&x| x - mean).collect::<Vec<_>>();
            let sum2: f32 = other.iter().map(|x| x * x).sum();
            let sum2 = sum2 / (lenf) + NORM_EPS;
            // let sum = sum2.sqrt(); // this is slower?
            // this is same speed as mutating the array normally, rustc is so much smarter now lmao
            let other = other.iter().map(|x| (x / sum2.sqrt()));

            Some(
                other
                    .zip(w.iter())
                    .zip(b.iter())
                    .map(|((x, &w), &b)| (x * w) + b)
                    .collect(),
            )
        }
    }
}

#[derive(Debug)]
pub(crate) struct Block<'a> {
    pub att: Att<'a>,
    pub ffn: Ffn<'a>,
    // pub ln0: Option<Ln<'a>>,
    pub ln1: Ln<'a>,
    pub ln2: Ln<'a>,
}

impl Block<'_> {
    pub fn apply(&self, x: &[f32], state: &mut StateElem) -> Vec<f32> {
        let xx = self.ln1.apply_f32(x).unwrap();
        let xx = self.att.apply(&xx, state);
        let x = x
            .iter()
            .zip(xx.iter())
            .map(|(&x, &y)| x + y)
            .collect::<Vec<_>>();

        let xx = self.ln2.apply_f32(&x).unwrap();
        let xx = self.ffn.apply(&xx, state);
        x.iter().zip(xx.iter()).map(|(&x, &y)| x + y).collect()
    }
}

#[derive(Debug)]
pub(crate) struct Emb<'a> {
    pub tensor: TensorView<'a>,
}

#[allow(dead_code)]
impl<'a> Emb<'a> {
    pub fn new(tensor: TensorView<'a>) -> Self {
        Self { tensor }
    }

    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    pub fn tokens(&self) -> usize {
        self.shape()[0]
    }

    pub fn nembed(&self) -> usize {
        self.shape()[1]
    }

    pub fn get(&self, row: usize) -> Option<&[bf16]> {
        if row > self.tokens() {
            None
        } else {
            Some(get_row_2d_bf16(&self.tensor, row))
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub(crate) struct EmbOptim {
    pub emb: Vec<Vec<f32>>,
    pub shape: [usize; 2],
}

impl EmbOptim {
    #[allow(unused_variables)]
    pub fn new(emb: &Emb, ln0: &Ln) -> Self {
        let shape = emb.shape();
        let emb = get_bf16(&emb.tensor)
            .chunks_exact(emb.nembed())
            .map(|x| x.iter().map(|&x| x.to_f32()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Self {
            emb,
            shape: [shape[0], shape[1]],
        }
    }

    pub fn get(&self, row: usize) -> Option<&[f32]> {
        if row > self.emb.len() {
            None
        } else {
            Some(&self.emb[row])
        }
    }
}

pub(crate) struct Rwkv<'a> {
    // pub emb: Emb<'a>,
    pub emb: EmbOptim,
    pub blocks: Vec<Block<'a>>,
    pub ln_out: Ln<'a>,
    pub head: TensorView<'a>,
}

impl Rwkv<'_> {
    pub fn forward_raw_preproc(&self, x: &[f32], state: &mut [StateElem]) -> Vec<f32> {
        let mut x = x.to_vec();
        for (i, state) in state.iter_mut().enumerate().take(self.blocks.len()) {
            x = self.blocks[i].apply(&x, state);
            if ((i + 1) % 6) == 0 {
                // x.iter_mut().for_each(|x| *x = *x / 2f32);
                for e in &mut x {
                    *e /= 2f32;
                }
            }
        }
        x
    }

    pub fn forward_raw(&self, x: &[f32], state: &mut [StateElem]) -> Vec<f32> {
        let xx = self.forward_raw_preproc(x, state);

        let xx = self.ln_out.apply_f32(&xx).unwrap();
        let h = get_bf16(&self.head);
        h.chunks_exact(xx.len())
            .map(|x| {
                x.iter()
                    .zip(xx.iter())
                    .map(|(&x, &y)| x.to_f32() * y)
                    .sum::<f32>()
            })
            .collect()
    }
}
