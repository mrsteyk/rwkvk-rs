use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    ptr::slice_from_raw_parts,
    time::Instant,
};

use anyhow::{Context, Result};
use clap::Parser;
use half::prelude::*;
use memmap2::Mmap;
use safetensors::tensor::{SafeTensors, TensorView};
use tokenizers::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    tokenizer: PathBuf,

    #[arg()]
    text: String,

    #[arg(short = 'p', long, default_value_t = 0f32)]
    alpha_presence: f32,

    #[arg(short = 'f', long, default_value_t = 0f32)]
    alpha_frequency: f32,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct StateElem {
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
struct Att<'a> {
    // Can be 16
    pub key: TensorView<'a>,
    pub output: TensorView<'a>,
    pub receptance: TensorView<'a>,
    // ---
    // Can't be 16
    // pub time_decay: TensorView<'a>,
    // pub time_first: TensorView<'a>,
    pub time_decay: &'a [f32],
    pub time_first: &'a [f32],
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
        let rw_len = self.receptance.shape()[0];
        let r = (0..rw_len)
            .map(|x| {
                (0..xr.len())
                    .map(|y| rw[xr.len() * x + y].to_f32() * xr[y])
                    .sum::<f32>()
            })
            // .map(|x| bf16::from_f32(1f32 / (1f32 - x.to_f32().exp())))
            .map(|x| (1f32 / (1f32 + (-x).exp())))
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        debug_assert_eq!(self.key.shape()[1], xk.len());
        let kw = get_bf16(&self.key);
        let kw_len = self.key.shape()[0];
        let k = (0..kw_len)
            .map(|x| {
                (0..xk.len())
                    .map(|y| kw[xk.len() * x + y].to_f32() * xk[y])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        debug_assert_eq!(self.value.shape()[1], xv.len());
        let vw = get_bf16(&self.value);
        let vw_len = self.value.shape()[0];
        let v = (0..vw_len)
            .map(|x| {
                (0..xv.len())
                    .map(|y| vw[xv.len() * x + y].to_f32() * xv[y])
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
        (0..rwkv.len())
            .map(|x| {
                (0..ow_len)
                    .map(|y| ow[ow_len * x + y].to_f32() * rwkv[y])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>()
    }
}

#[derive(Debug)]
struct Ffn<'a> {
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
        let rw_len = self.receptance.shape()[0];
        let r = (0..rw_len)
            .map(|x| {
                (0..xr.len())
                    .map(|y| rw[xr.len() * x + y].to_f32() * xr[y])
                    .sum::<f32>()
            })
            .map(|x| (1f32 / (1f32 + (-x).exp())))
            .collect::<Vec<_>>();

        #[cfg(debug_assertions)]
        debug_assert_eq!(self.key.shape()[1], xk.len());
        let kw = get_bf16(&self.key);
        let kw_len = self.key.shape()[0];
        // println!("{:?} {:?}", self.key.shape(), xk.len());
        let k = (0..kw_len)
            .map(|x| {
                (0..xk.len())
                    .map(|y| kw[xk.len() * x + y].to_f32() * xk[y])
                    .sum::<f32>()
            })
            .map(|x| if x.is_sign_positive() { x } else { 0f32 })
            .map(|x| x * x)
            .collect::<Vec<_>>();

        let vw = get_bf16(&self.value);
        let vw_len = self.value.shape()[0];
        let kv = (0..vw_len).map(|x| {
            (0..k.len())
                .map(|y| vw[k.len() * x + y].to_f32() * k[y])
                .sum::<f32>()
        });

        r.iter().zip(kv).map(|(&x, y)| x * y).collect()
    }
}

#[derive(Debug)]
struct Ln<'a> {
    pub weight: TensorView<'a>,
    pub bias: TensorView<'a>,
}

impl Ln<'_> {
    pub fn apply_bf16(&self, other: &[bf16]) -> Option<Vec<f32>> {
        if self.weight.shape()[0] != other.len() {
            None
        } else {
            // TODO: is converting to f32 worth it?
            const NORM_EPS: f32 = 1e5;

            // What the fuck
            let lenf = other.len() as f32;
            let mean: f32 = other.iter().map(|&x| x.to_f32()).sum();
            let mean = mean / lenf;
            let other = other.iter().map(|&x| x.to_f32() - mean);
            let sum2: f32 = other.clone().map(|x| x * x).sum();
            let sum2 = sum2 / (lenf) + NORM_EPS;
            let other = other.map(|x| (x / sum2.sqrt()));

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

    pub fn apply_f32(&self, other: &[f32]) -> Option<Vec<f32>> {
        if self.weight.shape()[0] != other.len() {
            None
        } else {
            // TODO: is converting to f32 worth it?
            const NORM_EPS: f32 = 1e-5;

            // What the fuck
            let lenf = other.len() as f32;
            let mean: f32 = other.iter().map(|&x| x).sum();
            let mean = mean / lenf;
            let other = other.iter().map(|&x| x - mean);
            let sum2: f32 = other.clone().map(|x| x * x).sum();
            let sum2 = sum2 / (lenf) + NORM_EPS;
            let other = other.map(|x| (x / sum2.sqrt()));

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
}

#[derive(Debug)]
struct Block<'a> {
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
struct Emb<'a> {
    pub tensor: TensorView<'a>,
}

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
struct EmbOptim {
    pub emb: Vec<Vec<f32>>,
    pub shape: [usize; 2],
}

impl EmbOptim {
    pub fn new(emb: &Emb, ln0: &Ln) -> Self {
        let shape = emb.shape();
        let emb = (0..emb.tokens())
            .map(|i| {
                let emb = emb.get(i).unwrap().to_vec();
                // ln0.apply_bf16(&emb).unwrap()
                emb.iter().map(|&x| x.to_f32()).collect()
            })
            .collect();
        Self {
            emb,
            shape: [shape[0], shape[1]],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn get(&self, row: usize) -> Option<&[f32]> {
        if row > self.emb.len() {
            None
        } else {
            Some(&self.emb[row])
        }
    }
}

// struct RWKV<'a, const T: usize> {
struct RWKV<'a> {
    // pub emb: Emb<'a>,
    pub emb: EmbOptim,
    pub blocks: Vec<Block<'a>>,
    pub ln_out: Ln<'a>,
    pub head: TensorView<'a>,
}

impl RWKV<'_> {
    pub fn forward_raw_preproc(&self, x: &[f32], state: &mut [StateElem]) -> Vec<f32> {
        let mut x = x.to_vec();
        for i in 0..self.blocks.len() {
            x = self.blocks[i].apply(&x, &mut state[i]);
            if ((i + 1) % 6) == 0 {
                // x.iter_mut().for_each(|x| *x = *x / 2f32);
                for i in 0..x.len() {
                    x[i] /= 2f32;
                }
            }
        }
        x
    }

    pub fn forward_raw(&self, x: &[f32], state: &mut [StateElem]) -> Vec<f32> {
        let xx = self.forward_raw_preproc(x, state);

        let xx = self.ln_out.apply_f32(&xx).unwrap();
        let h = get_bf16(&self.head);
        let h_len = self.head.shape()[0];
        (0..h_len)
            .map(|x| {
                (0..xx.len())
                    .map(|y| h[xx.len() * x + y].to_f32() * xx[y])
                    .sum::<f32>()
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let start = Instant::now();
    let tokenizer = Tokenizer::from_file(args.tokenizer).unwrap();
    // println!("{:?}", tokenizer.encode("<|endoftext|>", true));
    println!("Tokenizer load: {:?}", start.elapsed());

    let start = Instant::now();
    let model_file = File::open(args.model).context("error openning model file!")?;
    let map = unsafe { Mmap::map(&model_file).context("error mmaping model")? };
    let model = SafeTensors::deserialize(&map[..]).context("error parsing safetensor file")?;
    println!("SafeTensors parse: {:?}", start.elapsed());

    // println!("{:#?}", model.names());

    let start = Instant::now();
    let emb = Emb::new(
        model
            .tensor("emb.weight")
            .context("failed to get embedding")?,
    );
    println!("embedding: {:?}", emb.shape());

    // I decided to leave it here for the soft-emb
    let ln0 = Ln {
        weight: model.tensor("blocks.0.ln0.weight").unwrap(),
        bias: model.tensor("blocks.0.ln0.bias").unwrap(),
    };
    let optim_embed = Instant::now();
    let emb = { EmbOptim::new(&emb, &ln0) };
    let optim_embed = optim_embed.elapsed();
    println!("embedding optim: {:?}", optim_embed);
    // println!("embedding for 0 is: {:?}", emb.get(0).unwrap());
    // println!("ln0 embedding for 0 is: {:?}", ln0.apply_f32(emb.get(0).unwrap()));

    let blocks_num = {
        let mut r = 0usize;
        for i in model.names() {
            const PREFIX: &str = "blocks.";
            if i.starts_with(PREFIX) {
                let i = &i[PREFIX.len()..];
                let i = &i[..i.find('.').unwrap_or(0)];
                r = r.max(usize::from_str_radix(i, 10).context("error parsing blocks")?)
            }
        }
        r + 1
    };
    println!("blocks_num: {}", blocks_num);

    let blocks = (0..blocks_num)
        .map(|i| {
            // println!("Block {}", i);
            let time_decay = model
                .tensor(&format!("blocks.{}.att.time_decay", i))
                .unwrap();
            let time_first = model
                .tensor(&format!("blocks.{}.att.time_first", i))
                .unwrap();
            let time_decay = unsafe {
                &*slice_from_raw_parts(
                    time_decay.data().as_ptr().cast::<f32>(),
                    time_decay.shape()[0],
                )
            };
            let time_first = unsafe {
                &*slice_from_raw_parts(
                    time_first.data().as_ptr().cast::<f32>(),
                    time_first.shape()[0],
                )
            };

            let att = Att {
                key: model
                    .tensor(&format!("blocks.{}.att.key.weight", i))
                    .unwrap(),
                output: model
                    .tensor(&format!("blocks.{}.att.output.weight", i))
                    .unwrap(),
                receptance: model
                    .tensor(&format!("blocks.{}.att.receptance.weight", i))
                    .unwrap(),
                // ---
                time_decay,
                time_first,
                // ---
                time_mix_k: model
                    .tensor(&format!("blocks.{}.att.time_mix_k", i))
                    .unwrap(),
                time_mix_r: model
                    .tensor(&format!("blocks.{}.att.time_mix_r", i))
                    .unwrap(),
                time_mix_v: model
                    .tensor(&format!("blocks.{}.att.time_mix_v", i))
                    .unwrap(),
                value: model
                    .tensor(&format!("blocks.{}.att.value.weight", i))
                    .unwrap(),
            };

            let ffn = Ffn {
                key: model
                    .tensor(&format!("blocks.{}.ffn.key.weight", i))
                    .unwrap(),
                receptance: model
                    .tensor(&format!("blocks.{}.ffn.receptance.weight", i))
                    .unwrap(),
                time_mix_k: model
                    .tensor(&format!("blocks.{}.ffn.time_mix_k", i))
                    .unwrap(),
                time_mix_r: model
                    .tensor(&format!("blocks.{}.ffn.time_mix_r", i))
                    .unwrap(),
                value: model
                    .tensor(&format!("blocks.{}.ffn.value.weight", i))
                    .unwrap(),
            };

            let ln1 = Ln {
                weight: model.tensor(&format!("blocks.{}.ln1.weight", i)).unwrap(),
                bias: model.tensor(&format!("blocks.{}.ln1.bias", i)).unwrap(),
            };
            let ln2 = Ln {
                weight: model.tensor(&format!("blocks.{}.ln2.weight", i)).unwrap(),
                bias: model.tensor(&format!("blocks.{}.ln2.bias", i)).unwrap(),
            };

            Block { att, ffn, ln1, ln2 }
        })
        .collect::<Vec<_>>();

    let head = model.tensor("head.weight").unwrap();
    let ln_out = Ln {
        weight: model.tensor("ln_out.weight").unwrap(),
        bias: model.tensor("ln_out.bias").unwrap(),
    };

    let rwkv = RWKV {
        emb,
        blocks,
        head,
        ln_out,
    };
    let load = start.elapsed();
    println!("RWKV load: {:?} ({:?})", load, load - optim_embed);

    let start = Instant::now();
    let tokens = tokenizer
        .encode(args.text.clone(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    println!("Tokenization: {:?} {:?}", start.elapsed(), &tokens);

    let start = Instant::now();
    let mut state = vec![
        StateElem {
            ffn_x: vec![0f32; rwkv.emb.shape[1]],
            att_x: vec![0f32; rwkv.emb.shape[1]],
            att_a: vec![0f32; rwkv.emb.shape[1]],
            att_b: vec![0f32; rwkv.emb.shape[1]],
            att_p: vec![-1e30; rwkv.emb.shape[1]],
        };
        rwkv.blocks.len()
    ];
    for &i in &tokens[..tokens.len() - 1] {
        let e = rwkv.emb.get(i as usize).unwrap();
        rwkv.forward_raw_preproc(e, &mut state);
    }

    let mut x = rwkv.forward_raw(
        rwkv.emb.get(tokens[tokens.len() - 1] as usize).unwrap(),
        &mut state,
    );
    println!("Preprocess: {:?}", start.elapsed());

    let mut alpha = vec![0usize; rwkv.emb.shape[0]];

    print!("{}", &args.text);
    std::io::stdout().flush().unwrap();
    for _ in 0..767 {
        let (token, token_weight) = x
            .iter()
            .enumerate()
            .map(|(ei, &e)| {
                (
                    ei,
                    e - (alpha[ei] as f32 * args.alpha_frequency)
                        - (if alpha[ei] > 0 {
                            args.alpha_presence
                        } else {
                            0f32
                        }),
                )
            })
            .fold(
                (0, f32::MIN),
                |(mi, me), (ei, e)| if e > me { (ei, e) } else { (mi, me) },
            );

        // EOS
        if token == 0 {
            break;
        }

        alpha[token] += 1;

        // print!("{}[{};{}]", tokenizer.id_to_token(token as u32).unwrap_or("<UNK>".to_string()), token, token_weight);
        print!(
            "{}",
            tokenizer
                .decode(vec![token as u32], false)
                .unwrap_or("<UNK>".to_string())
        );
        std::io::stdout().flush().unwrap();

        x = rwkv.forward_raw(rwkv.emb.get(token).unwrap(), &mut state);
    }

    println!("");
    Ok(())
}

fn get_row_2d_bf16<'a>(emb: &TensorView<'a>, row: usize) -> &'a [bf16] {
    let len = emb.shape()[1];
    let row_size = len * 2;
    let idx = row_size * row;
    unsafe { &*slice_from_raw_parts(emb.data()[idx..idx + row_size].as_ptr().cast::<bf16>(), len) }
}

fn get_bf16<'a>(tensor: &TensorView<'a>) -> &'a [bf16] {
    unsafe {
        &*slice_from_raw_parts(
            tensor.data().as_ptr().cast::<bf16>(),
            tensor.shape().iter().fold(1, |a, &x| a * x),
        )
    }
}
