use half::bf16;
use safetensors::tensor::TensorView;

pub fn get_row_2d_bf16<'a>(emb: &TensorView<'a>, row: usize) -> &'a [bf16] {
    let len = emb.shape()[1];
    let row_size = len * 2;
    let idx = row_size * row;
    bytemuck::cast_slice(&emb.data()[idx..idx + row_size])
}

pub fn get_bf16<'a>(tensor: &TensorView<'a>) -> &'a [bf16] {
    bytemuck::cast_slice(tensor.data())
}
