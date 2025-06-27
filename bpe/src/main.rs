mod dataset;
mod bpe;
mod common;
mod pretokenize;
use std::collections::HashMap;
use std::time::Instant;
use crate::dataset::get_dataset;
use crate::bpe::{ encode, decode, train };

fn main() {
    let now = Instant::now();
    let text = get_dataset();
    let text_ref: &str = &text;

    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut vocab: HashMap<u32, String> = HashMap::new();

    let ids = text.as_bytes();

    let (u32_ids, _vocab_bytes) = train(text_ref, 512, vec!["<|endoftext|>".to_string()], &mut merges, &mut vocab, &mut counts, false);

    let elapsed = now.elapsed();
    println!("Training Elapsed: {:.2?}", elapsed);

    let encoding = encode("I am loved by many of my followers", &merges);
    print!("\nEncoded Value: ");
    for item in encoding.iter() {
        print!("{},", item);
    }

    let decoding = decode(&encoding, &vocab);
    println!("\nDecoding: {}", decoding);

    println!(
        "Compression {}/{} = {:.3}x",
        ids.len(),
        u32_ids.len(),
        (ids.len() as f32) / (u32_ids.len() as f32)
    );
}
