use std::{marker::PhantomPinned, pin::Pin, ptr::NonNull, fs::File, mem::MaybeUninit};

use anyhow::{Result, Context};
use memmap2::Mmap;
pub use safetensors;

mod rwkv;
mod utils;

pub use rwkv::*;
use safetensors::tensor::SafeTensors;

impl<'a> Rwkv<'a> {
    pub fn from_safetensors(model: &'a SafeTensors) -> Result<Self> {
        let emb = rwkv::Emb::new(
            model
                .tensor("emb.weight")
                .context("failed to get embedding")?,
        );

        // I decided to leave it here for the soft-emb
        let ln0 = rwkv::Ln {
            weight: model.tensor("blocks.0.ln0.weight").context("failed to get blocks.0.ln0.weight")?,
            bias: model.tensor("blocks.0.ln0.bias").context("failed to get blocks.0.ln0.bias")?,
        };
        let emb = { rwkv::EmbOptim::new(&emb, &ln0) };

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

        let head = model.tensor("head.weight").context("error getting head.weight")?;
        let ln_out = rwkv::Ln {
            weight: model.tensor("ln_out.weight").context("error getting ln_out.weight")?,
            bias: model.tensor("ln_out.bias").context("error getting ln_out.bias")?,
        };

        let mut blocks = Vec::with_capacity(blocks_num);
        for i in 0..blocks_num {
            let time_decay = model.tensor(&format!("blocks.{i}.att.time_decay")).context("error getting time_decay")?;
                let time_first = model.tensor(&format!("blocks.{i}.att.time_first")).context("error getting time_first")?;
                let time_decay = bytemuck::pod_collect_to_vec::<_, f32>(time_decay.data());
                let time_first = bytemuck::pod_collect_to_vec::<_, f32>(time_first.data());

                let att = rwkv::Att {
                    key: model.tensor(&format!("blocks.{i}.att.key.weight")).context("error getting att.key.weight")?,
                    output: model
                        .tensor(&format!("blocks.{i}.att.output.weight"))
                        .context("error getting att.output.weight")?,
                    receptance: model
                        .tensor(&format!("blocks.{i}.att.receptance.weight"))
                        .context("error getting att.receptance.weight")?,
                    // ---
                    time_decay,
                    time_first,
                    // ---
                    time_mix_k: model.tensor(&format!("blocks.{i}.att.time_mix_k")).context("error getting att.time_mix_k")?,
                    time_mix_r: model.tensor(&format!("blocks.{i}.att.time_mix_r")).context("error getting att.time_mix_r")?,
                    time_mix_v: model.tensor(&format!("blocks.{i}.att.time_mix_v")).context("error getting att.time_mix_v")?,
                    value: model
                        .tensor(&format!("blocks.{i}.att.value.weight"))
                        .context("error getting time_decay")?,
                };

                let ffn = rwkv::Ffn {
                    key: model.tensor(&format!("blocks.{i}.ffn.key.weight")).context("error getting ffn.key.weight")?,
                    receptance: model
                        .tensor(&format!("blocks.{i}.ffn.receptance.weight"))
                        .context("error getting ffn.receptance.weight")?,
                    time_mix_k: model.tensor(&format!("blocks.{i}.ffn.time_mix_k")).context("error getting ffn.time_mix_k")?,
                    time_mix_r: model.tensor(&format!("blocks.{i}.ffn.time_mix_r")).context("error getting ffn.time_mix_r")?,
                    value: model
                        .tensor(&format!("blocks.{i}.ffn.value.weight"))
                        .context("error getting time_decay")?,
                };

                let ln1 = rwkv::Ln {
                    weight: model.tensor(&format!("blocks.{i}.ln1.weight")).context("error getting ln1.weight")?,
                    bias: model.tensor(&format!("blocks.{i}.ln1.bias")).context("error getting ln1.bias")?,
                };
                let ln2 = rwkv::Ln {
                    weight: model.tensor(&format!("blocks.{i}.ln2.weight")).context("error getting ln2.weight")?,
                    bias: model.tensor(&format!("blocks.{i}.ln2.bias")).context("error getting ln2.bias")?,
                };

                blocks.push(Block { att, ffn, ln1, ln2 })
        }

        Ok(rwkv::Rwkv {
            emb,
            ln0: Some(ln0),
            blocks,
            head,
            ln_out,
        })
    }
}

const DUMMY: &[u8; 65] = b"8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}\x00";

pub struct RwkvWrap<'a> {
    rwkv: Option<Rwkv<'a>>,
    model: SafeTensors<'a>,
    map: Mmap,
    _pin: PhantomPinned,
}

impl<'a> RwkvWrap<'a> {
    pub fn rwkv(&self) -> &Rwkv<'a> {
        self.rwkv.as_ref().unwrap()
    }

    pub fn new_from_buf(buf: &'a [u8]) -> Result<Pin<Box<Self>>> {
        let model = SafeTensors::deserialize(buf)?;
        let ret = Self {
            model,
            // Blergh...
            map: unsafe { MaybeUninit::zeroed().assume_init() },
            rwkv: None,
            _pin: PhantomPinned
        };

        // Welcome to Rust!
        let mut boxed = Box::pin(ret);

        let model_ref = NonNull::from(&boxed.model);
        unsafe {
            let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
            Pin::get_unchecked_mut(mut_ref).rwkv = Some(Rwkv::from_safetensors(model_ref.as_ref())?);
        }

        Ok(boxed)
    }

    pub fn new_from_path(path: &str) -> Result<Pin<Box<Self>>> {
        let model_file = File::open(path).context("error openning model file!")?;
        let map = unsafe { Mmap::map(&model_file).context("error mmaping model")? };
        // let model = SafeTensors::deserialize(&map[..])?;
        let ret = Self {
            model: SafeTensors::deserialize(DUMMY).unwrap(),
            map,
            rwkv: None,
            _pin: PhantomPinned,
        };

        // Welcome to Rust!
        let mut boxed = Box::pin(ret);

        let map_ref = NonNull::from(&boxed.map);
        unsafe {
            let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
            Pin::get_unchecked_mut(mut_ref).model = SafeTensors::deserialize(&map_ref.as_ref()[..])?;
        }

        let model_ref = NonNull::from(&boxed.model);
        unsafe {
            let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
            Pin::get_unchecked_mut(mut_ref).rwkv = Some(Rwkv::from_safetensors(model_ref.as_ref())?);
        }

        Ok(boxed)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_deserialization() {
        safetensors::tensor::SafeTensors::deserialize(super::DUMMY).unwrap();
    }

    #[test]
    fn test_loading_path_please_remove() {
        let model = super::RwkvWrap::new_from_path("../../RWKV-LM-deepspeed/RWKV-v4neo/RWKV-4-Pile-430M-20220808-8066.rnn.bf16.safetensors").unwrap();
        let mut state = vec![
            super::StateElem {
                ffn_x: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_x: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_a: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_b: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_p: vec![-1e30; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
            };
            model.as_ref().rwkv.as_ref().unwrap().blocks.len()
        ];
        model.as_ref().rwkv.as_ref().unwrap().forward_raw_preproc(&[0f32; 1024], &mut state);
    }

    #[test]
    fn test_loading_buf_please_remove() {
        let model_file = super::File::open("../../RWKV-LM-deepspeed/RWKV-v4neo/RWKV-4-Pile-430M-20220808-8066.rnn.bf16.safetensors").unwrap();
        let map = unsafe { super::Mmap::map(&model_file).unwrap() };
        let model = super::RwkvWrap::new_from_buf(&map[..]).unwrap();
        // We are not alowed to drop map because it is borrowed for RwkvWrap, works as intented!
        // drop(map);
        let mut state = vec![
            super::StateElem {
                ffn_x: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_x: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_a: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_b: vec![0f32; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
                att_p: vec![-1e30; model.as_ref().rwkv.as_ref().unwrap().emb.shape[1]],
            };
            model.as_ref().rwkv.as_ref().unwrap().blocks.len()
        ];
        model.as_ref().rwkv.as_ref().unwrap().forward_raw_preproc(&[0f32; 1024], &mut state);
    }
}