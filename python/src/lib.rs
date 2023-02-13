use std::{pin::Pin, sync::Arc};

use pyo3::{prelude::*, types::PyList};

#[derive(Debug)]
enum RwkvError {
    InvalidToken(usize),
    EmptyArray,
}

impl std::fmt::Display for RwkvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RwkvError::InvalidToken(n) => f.write_fmt(format_args!("Invalid Token: {:?}", n)),
            RwkvError::EmptyArray => f.write_str("Empty array"),
        }
    }
}

impl std::error::Error for RwkvError {
    // ???
}

#[pyclass]
struct Rwkv {
    inner: Arc<Pin<Box<rwkvk_rs::RwkvWrap<'static>>>>,
}

#[pyclass]
struct State {
    inner: Vec<rwkvk_rs::StateElem>,
}

#[pymethods]
impl State {
    #[new]
    pub fn new(model: &Rwkv) -> PyResult<Self> {
        let model = model.inner.as_ref().as_ref();
        let model = model.rwkv();
        let state = vec![
            rwkvk_rs::StateElem {
                ffn_x: vec![0f32; model.emb.shape[1]],
                att_x: vec![0f32; model.emb.shape[1]],
                att_a: vec![0f32; model.emb.shape[1]],
                att_b: vec![0f32; model.emb.shape[1]],
                att_p: vec![-1e30; model.emb.shape[1]],
            };
            model.blocks.len()
        ];
        Ok(Self { inner: state })
    }
}

#[pymethods]
impl Rwkv {
    #[new]
    pub fn from_path(path: &str) -> PyResult<Self> {
        Ok(Self { inner: Arc::new(rwkvk_rs::RwkvWrap::new_from_path(path)?) })
    }

    pub fn forward_raw_preproc(&self, x: &PyList, state: &mut State) -> PyResult<()> {
        let x: Vec<f32> = x.extract()?;
        self.inner.rwkv().forward_raw_preproc(&x, &mut state.inner);
        Ok(())
    }

    pub fn forward_raw(&self, x: &PyList, state: &mut State) -> PyResult<Vec<f32>> {
        let x: Vec<f32> = x.extract()?;
        Ok(self.inner.rwkv().forward_raw(&x, &mut state.inner))
    }

    pub fn forward_raw_batch(&self, tokens: &PyList, state: &mut State) -> PyResult<Vec<f32>> {
        let tokens: Vec<Vec<f32>> = tokens.extract()?;
        if tokens.len() == 0 {
            return Err(anyhow::anyhow!(RwkvError::EmptyArray).into());
        }
        for token in tokens.iter().take(tokens.len()-1) {
            self.inner.rwkv().forward_raw_preproc(token, &mut state.inner);
        }
        Ok(self.inner.rwkv().forward_raw(tokens.last().unwrap(), &mut state.inner))
    }

    pub fn forward_token(&self, token: usize, state: &mut State) -> PyResult<Vec<f32>> {
        let x = self.inner.rwkv().emb.get(token).ok_or(anyhow::anyhow!(RwkvError::InvalidToken(token)))?;
        Ok(self.inner.rwkv().forward_raw(x, &mut state.inner))
    }

    pub fn forward(&self, tokens: &PyList, state: &mut State) -> PyResult<Vec<f32>> {
        let tokens: Vec<usize> = tokens.extract()?;
        if tokens.len() == 0 {
            return Err(anyhow::anyhow!(RwkvError::EmptyArray).into());
        }
        for &token in tokens.iter().take(tokens.len()-1) {
            let token = self.inner.rwkv().emb.get(token).ok_or(anyhow::anyhow!(RwkvError::InvalidToken(token)))?;
            self.inner.rwkv().forward_raw_preproc(token, &mut state.inner);
        }
        let token = *tokens.last().unwrap();
        let token = self.inner.rwkv().emb.get(token).ok_or(anyhow::anyhow!(RwkvError::InvalidToken(token)))?;
        Ok(self.inner.rwkv().forward_raw(token, &mut state.inner))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rwkv_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Rwkv>()?;
    m.add_class::<State>()?;
    Ok(())
}
