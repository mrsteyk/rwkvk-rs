# RWKV, but in Rust...

This is pretty much your usual RWKV, but it follows the RIIR principle.

Notable tech stack uses are [memmap2](https://github.com/RazrFalcon/memmap2-rs) and [safetensors](https://github.com/huggingface/safetensors).

# FAQ

### How to install?

This can break and if it does use `--rev` or `--tag` flags to specify which version to install.

```sh
cargo +nightly install --git https://github.com/mrsteyk/rwkvk-rs --features="build-binary"
```

### How do I get a model for this?

Either download one from [hf:mrsteyk/RWKV-LM-safetensors](https://huggingface.co/mrsteyk/RWKV-LM-safetensors/tree/main) or convert yourself using `convert_safetensors.py` from [gh:mrsteyk/RWKV-LM-deepspeed](https://github.com/mrsteyk/RWKV-LM-deepspeed) with arguments `--bf16 --rnn`. Make sure to read the next section about alignment.

### I get some weird errors when I try to use models I converted myself?

See [gh:huggingface/safetensors#178](https://github.com/huggingface/safetensors/issues/178).

TL;DR you need to pad the header, no need to reorder tensors because I copy `time_` tensors (for now at least).
