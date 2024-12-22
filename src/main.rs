use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{Embedding, VarBuilder};

fn embedding(num_embeddings: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((num_embeddings, embedding_dim), "weight")?;
    Ok(Embedding::new(embeddings, embedding_dim))
}

struct LearnedPositionEmbeddings {
    emb: Embedding,
    relative: bool,
    seq_len: usize,
}

fn main() {
    println!("Hello, world!");
}
