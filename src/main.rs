use std::{fs::File, io::Write, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use half::prelude::*;
use memmap2::Mmap;
use safetensors::tensor::{SafeTensors, TensorView};
use tokenizers::tokenizer::Tokenizer;

pub mod rwkv;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    tokenizer: PathBuf,

    #[arg(short, long)]
    soft_emb: Option<PathBuf>,

    #[arg()]
    text: String,

    #[arg(short = 'p', long, default_value_t = 0f32)]
    alpha_presence: f32,

    #[arg(short = 'f', long, default_value_t = 0f32)]
    alpha_frequency: f32,
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
    let emb = rwkv::Emb::new(
        model
            .tensor("emb.weight")
            .context("failed to get embedding")?,
    );
    println!("embedding: {:?}", emb.shape());

    // I decided to leave it here for the soft-emb
    let ln0 = rwkv::Ln {
        weight: model.tensor("blocks.0.ln0.weight").unwrap(),
        bias: model.tensor("blocks.0.ln0.bias").unwrap(),
    };
    let optim_embed = Instant::now();
    let emb = { rwkv::EmbOptim::new(&emb, &ln0) };
    let optim_embed = optim_embed.elapsed();
    println!("embedding optim: {optim_embed:?}");
    // println!("embedding for 0 is: {:?}", emb.get(0).unwrap());
    // println!("ln0 embedding for 0 is: {:?}", ln0.apply_f32(emb.get(0).unwrap()));

    let blocks_num = {
        let mut r = 0usize;
        for i in model.names() {
            const PREFIX: &str = "blocks.";
            if let Some(i) = i.strip_prefix(PREFIX) {
                // let i = &i[PREFIX.len()..];
                let i = &i[..i.find('.').unwrap_or(0)];
                r = r.max(i.parse::<usize>().context("error parsing blocks")?)
            }
        }
        r + 1
    };
    println!("blocks_num: {blocks_num}");

    let blocks = (0..blocks_num)
        .map(|i| {
            // println!("Block {}", i);
            let time_decay = model.tensor(&format!("blocks.{i}.att.time_decay")).unwrap();
            let time_first = model.tensor(&format!("blocks.{i}.att.time_first")).unwrap();
            let time_decay = bytemuck::pod_collect_to_vec::<_, f32>(time_decay.data());
            let time_first = bytemuck::pod_collect_to_vec::<_, f32>(time_first.data());

            let att = rwkv::Att {
                key: model.tensor(&format!("blocks.{i}.att.key.weight")).unwrap(),
                output: model
                    .tensor(&format!("blocks.{i}.att.output.weight"))
                    .unwrap(),
                receptance: model
                    .tensor(&format!("blocks.{i}.att.receptance.weight"))
                    .unwrap(),
                // ---
                time_decay,
                time_first,
                // ---
                time_mix_k: model.tensor(&format!("blocks.{i}.att.time_mix_k")).unwrap(),
                time_mix_r: model.tensor(&format!("blocks.{i}.att.time_mix_r")).unwrap(),
                time_mix_v: model.tensor(&format!("blocks.{i}.att.time_mix_v")).unwrap(),
                value: model
                    .tensor(&format!("blocks.{i}.att.value.weight"))
                    .unwrap(),
            };

            let ffn = rwkv::Ffn {
                key: model.tensor(&format!("blocks.{i}.ffn.key.weight")).unwrap(),
                receptance: model
                    .tensor(&format!("blocks.{i}.ffn.receptance.weight"))
                    .unwrap(),
                time_mix_k: model.tensor(&format!("blocks.{i}.ffn.time_mix_k")).unwrap(),
                time_mix_r: model.tensor(&format!("blocks.{i}.ffn.time_mix_r")).unwrap(),
                value: model
                    .tensor(&format!("blocks.{i}.ffn.value.weight"))
                    .unwrap(),
            };

            let ln1 = rwkv::Ln {
                weight: model.tensor(&format!("blocks.{i}.ln1.weight")).unwrap(),
                bias: model.tensor(&format!("blocks.{i}.ln1.bias")).unwrap(),
            };
            let ln2 = rwkv::Ln {
                weight: model.tensor(&format!("blocks.{i}.ln2.weight")).unwrap(),
                bias: model.tensor(&format!("blocks.{i}.ln2.bias")).unwrap(),
            };

            rwkv::Block { att, ffn, ln1, ln2 }
        })
        .collect::<Vec<_>>();

    let head = model.tensor("head.weight").unwrap();
    let ln_out = rwkv::Ln {
        weight: model.tensor("ln_out.weight").unwrap(),
        bias: model.tensor("ln_out.bias").unwrap(),
    };

    let rwkv = rwkv::Rwkv {
        emb,
        blocks,
        head,
        ln_out,
    };
    let load = start.elapsed();
    println!(
        "RWKV load: {:?} ({:?} {:?})",
        load,
        optim_embed,
        load - optim_embed
    );

    let start = Instant::now();
    let tokens = tokenizer
        .encode(args.text.clone(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    println!("Tokenization: {:?} {:?}", start.elapsed(), &tokens);

    let mut state = vec![
        rwkv::StateElem {
            ffn_x: vec![0f32; rwkv.emb.shape[1]],
            att_x: vec![0f32; rwkv.emb.shape[1]],
            att_a: vec![0f32; rwkv.emb.shape[1]],
            att_b: vec![0f32; rwkv.emb.shape[1]],
            att_p: vec![-1e30; rwkv.emb.shape[1]],
        };
        rwkv.blocks.len()
    ];
    let start = Instant::now();
    if let Some(soft_emb) = args.soft_emb {
        let model_file = File::open(soft_emb).context("error openning model file!")?;
        let map = unsafe { Mmap::map(&model_file).context("error mmaping model")? };
        let model = SafeTensors::deserialize(&map[..]).context("error parsing safetensor file")?;

        let soft = &model.tensors()[0].1;
        // This is not that much faster... idk why LN takes so much time... perhaps all that conversion between bf16 and f32...
        // let w = get_bf16(&ln0.weight).iter().map(|&x| x.to_f32()).collect::<Vec<_>>();
        // let b = get_bf16(&ln0.bias).iter().map(|&x| x.to_f32()).collect::<Vec<_>>();
        // for i in 0..soft.shape()[0] {
        //     rwkv.forward_raw_preproc(
        //         // &Ln::apply_f32_f(
        //         // F32 is faster by like 0.05s??? F32+ln0.F32 is faster than F32 by like 0.01s??? what the fuck did I do to make it so slow
        //         &match soft.dtype() {
        //             safetensors::tensor::Dtype::BF16 => ln0.apply_f32(
        //                 &get_row_2d_bf16(soft, i)
        //                     .iter()
        //                     .map(|&x| x.to_f32())
        //                     .collect::<Vec<_>>(),
        //             ),
        //             safetensors::tensor::Dtype::F32 => {
        //                 let len = soft.shape()[1];
        //                 let size = len * 4;
        //                 ln0.apply_f32(unsafe {
        //                     &*slice_from_raw_parts(
        //                         soft.data()[size * i..size * i + size]
        //                             .as_ptr()
        //                             .cast::<f32>(),
        //                         len,
        //                     )
        //                 })
        //             }
        //             _ => unreachable!(),
        //         }
        //         .unwrap(),
        //         &mut state,
        //     );
        // }
        // ---
        // This is somehow slower??? But I don't think benchmarks are fair because I am at the liberty of OS kernel
        let w = get_bf16(&ln0.weight)
            .iter()
            .map(|&x| x.to_f32())
            .collect::<Vec<_>>();
        let b = get_bf16(&ln0.bias)
            .iter()
            .map(|&x| x.to_f32())
            .collect::<Vec<_>>();
        match soft.dtype() {
            safetensors::tensor::Dtype::BF16 => get_bf16(soft)
                .chunks_exact(soft.shape()[1])
                .map(|x| {
                    rwkv::Ln::apply_f32_f(
                        &x.iter().map(|&x| x.to_f32()).collect::<Vec<_>>(),
                        &w,
                        &b,
                    )
                    .unwrap()
                })
                .for_each(|x| {
                    rwkv.forward_raw_preproc(&x, &mut state);
                }),
            // Small performance penalty?
            safetensors::tensor::Dtype::F32 => bytemuck::cast_slice::<_, f32>(&soft.data().to_vec())
                .chunks_exact(soft.shape()[1])
                .for_each(|x| {
                    rwkv.forward_raw_preproc(x, &mut state);
                }),
            _ => unreachable!(),
        }
    }
    let soft_elapsed = start.elapsed();
    for &i in &tokens[..tokens.len() - 1] {
        let e = rwkv.emb.get(i as usize).unwrap();
        rwkv.forward_raw_preproc(e, &mut state);
    }

    let mut x = rwkv.forward_raw(
        rwkv.emb.get(tokens[tokens.len() - 1] as usize).unwrap(),
        &mut state,
    );
    let preproc = start.elapsed();
    println!(
        "Preprocess: {:?} ({:?} {:?})",
        preproc,
        soft_elapsed,
        preproc - soft_elapsed
    );

    let mut alpha = vec![0usize; rwkv.emb.shape[0]];

    print!("{}", &args.text);
    std::io::stdout().flush().unwrap();
    for _ in 0..767 {
        #[allow(unused_variables)]
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

    println!();
    Ok(())
}

fn get_row_2d_bf16<'a>(emb: &TensorView<'a>, row: usize) -> &'a [bf16] {
    let len = emb.shape()[1];
    let row_size = len * 2;
    let idx = row_size * row;
    bytemuck::cast_slice(&emb.data()[idx..idx + row_size])
}

fn get_bf16<'a>(tensor: &TensorView<'a>) -> &'a [bf16] {
    bytemuck::cast_slice(tensor.data())
}
