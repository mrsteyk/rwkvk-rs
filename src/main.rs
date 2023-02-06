use std::{
    fs::File,
    path::{Path, PathBuf},
    ptr::slice_from_raw_parts,
};

use anyhow::{Context, Result};
use clap::Parser;
use half::bf16;
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
}

struct Head<'a> {
    pub weight: TensorView<'a>,
}

impl<'a> Head<'a> {
    pub fn new(weight: TensorView<'a>) -> Self {
        Self { weight }
    }

    pub fn shape(&self) -> &[usize] {
        self.weight.shape()
    }

    pub fn get(&self, row: usize) -> Option<&[bf16]> {
        if row > self.shape()[0] {
            None
        } else {
            Some(get_row_2d_bf16(&self.weight, row))
        }
    }
}

struct Att<'a> {
    // Can be 16
    pub key: TensorView<'a>,
    pub output: TensorView<'a>,
    pub receptence: TensorView<'a>,
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

struct Ffn<'a> {
    // Can be 16
    pub key: TensorView<'a>,
    pub receptence: TensorView<'a>,
    // ---
    pub time_mix_k: TensorView<'a>,
    pub time_mix_r: TensorView<'a>,
    // ---
    pub value: TensorView<'a>,
}

struct Ln<'a> {
    pub weight: TensorView<'a>,
    pub bias: TensorView<'a>,
}

struct Block<'a> {
    pub att: Att<'a>,
    pub ffn: Ffn<'a>,
    pub ln0: Option<Ln<'a>>,
    pub ln1: Ln<'a>,
    pub ln2: Ln<'a>,
}

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

// struct RWKV<'a, const T: usize> {
struct RWKV<'a> {
    pub emb: Emb<'a>,
    pub blocks: Vec<Block<'a>>,
    pub head: Head<'a>,
    pub ln_out: Ln<'a>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tokenizer = Tokenizer::from_file(args.tokenizer).unwrap();
    // println!("{:?}", tokenizer.encode("<|endoftext|>", true));

    let model_file = File::open(args.model).context("error openning model file!")?;
    let map = unsafe { Mmap::map(&model_file).context("error mmaping model")? };
    let model = SafeTensors::deserialize(&map[..]).context("error parsing safetensor file")?;

    // println!("{:#?}", model.names());

    let emb = Emb::new(
        model
            .tensor("emb.weight")
            .context("failed to get embedding")?,
    );
    println!("embedding: {:?}", emb.shape());

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

    // let emb0 = get_row_2d_bf16(emb, 0);
    // println!("emb0: {:?}", emb0);

    let blocks = (0..blocks_num)
        .map(|i| {
            // let base = format!("blocks.{}.", i);
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
                receptence: model
                    .tensor(&format!("blocks.{}.att.receptence.weight", i))
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
                receptence: model
                    .tensor(&format!("blocks.{}.ffn.receptence.weight", i))
                    .unwrap(),
                time_mix_k: model
                    .tensor(&format!("blocks.{}.ffn.time_mix_k", i))
                    .unwrap(),
                time_mix_r: model
                    .tensor(&format!("blocks.{}.ffn.time_mix_k", i))
                    .unwrap(),
                value: model
                    .tensor(&format!("blocks.{}.ffn.value.weight", i))
                    .unwrap(),
            };

            let ln0 = if let Ok(ln0_weight) = model.tensor(&format!("blocks.{}.ln0.weight", i)) {
                Some(Ln {
                    weight: ln0_weight,
                    bias: model.tensor(&format!("blocks.{}.ln0.bias", i)).unwrap(),
                })
            } else {
                None
            };
            let ln1 = Ln {
                weight: model.tensor(&format!("blocks.{}.ln1.weight", i)).unwrap(),
                bias: model.tensor(&format!("blocks.{}.ln1.bias", i)).unwrap(),
            };
            let ln2 = Ln {
                weight: model.tensor(&format!("blocks.{}.ln2.weight", i)).unwrap(),
                bias: model.tensor(&format!("blocks.{}.ln2.bias", i)).unwrap(),
            };

            Block {
                att,
                ffn,
                ln0,
                ln1,
                ln2
            }
        })
        .collect::<Vec<_>>();

    Ok(())
}

fn get_row_2d_bf16<'a>(emb: &safetensors::tensor::TensorView<'a>, row: usize) -> &'a [bf16] {
    let len = emb.shape()[1];
    let row_size = len * 2;
    let idx = row_size * row;
    unsafe { &*slice_from_raw_parts(emb.data()[idx..idx + row_size].as_ptr().cast::<bf16>(), len) }
}
