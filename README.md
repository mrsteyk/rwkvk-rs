# RWKV, but in Rust

This is pretty much your usual RWKV, but it follows the RIIR principle.

Notable tech stack uses are [memmap2](https://github.com/RazrFalcon/memmap2-rs) and [safetensors](https://github.com/huggingface/safetensors).

## FAQ

### Are there other projects?

Yes!

- <https://github.com/KerfuffleV2/smolrsrwkv>

### How to install?

This can break and if it does use `--rev` or `--tag` flags to specify which version to install.

```sh
cargo +nightly install --git https://github.com/mrsteyk/rwkvk-rs --features="build-binary"
```

### How to run?

Example:

```sh
cargo run --features build-binary --release -- -t ../../RWKV-LM-deepspeed/20B_tokenizer_openchatgpt.json -m ../../RWKV-LM-deepspeed/RWKV-v4neo/RWKV-4-Pile-430M-20220808-8066.rnn.bf16.safetensors "hi!"
```

### How do I get a model for this?

Either download one from [hf:mrsteyk/RWKV-LM-safetensors](https://huggingface.co/mrsteyk/RWKV-LM-safetensors/tree/main) or convert yourself using `convert_safetensors.py` from [gh:mrsteyk/RWKV-LM-deepspeed](https://github.com/mrsteyk/RWKV-LM-deepspeed) with arguments `--bf16 --rnn`. Make sure to read the next section about alignment.

### I get some weird errors when I try to use models I converted myself?

#### UPDATE

Update your `safetensors` version! [gh:huggingface/safetensors#148](https://github.com/huggingface/safetensors/pull/148) got merged!

#### OLD (kept for historic purposes)

See [gh:huggingface/safetensors#178](https://github.com/huggingface/safetensors/issues/178).

TL;DR you need to pad the header, no need to reorder tensors because I copy `time_` tensors (for now at least).
