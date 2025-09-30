use anyhow::{Result, anyhow};
use std::collections::HashMap;

pub const BYTE_VOCAB_SIZE: usize = 256;

#[derive(Debug, Clone, Copy)]
pub enum EncodeInput<'a> {
    Text(&'a str),
    Special(&'a str),
}

pub struct Tokenizer {
    special_to_id: HashMap<String, u32>,
    id_to_special: Vec<String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            special_to_id: HashMap::new(),
            id_to_special: Vec::new(),
        }
    }

    pub fn with_special_tokens<I>(tokens: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut tokenizer = Self::new();
        tokenizer.add_special_tokens(tokens)?;
        Ok(tokenizer)
    }

    pub fn vocab_size(&self) -> usize {
        BYTE_VOCAB_SIZE + self.id_to_special.len()
    }

    pub fn add_special_tokens<I>(&mut self, tokens: I) -> Result<Vec<u32>>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut ids = Vec::new();
        for token in tokens {
            let token = token.as_ref();
            if token.is_empty() {
                return Err(anyhow!("Special token cannot be empty"));
            }
            if let Some(&id) = self.special_to_id.get(token) {
                ids.push(id);
                continue;
            }
            if token.as_bytes().iter().any(|b| *b == b'\0') {
                return Err(anyhow!("Special tokens cannot contain null bytes"));
            }
            let id = (BYTE_VOCAB_SIZE + self.id_to_special.len())
                .try_into()
                .map_err(|_| anyhow!("Vocabulary exceeded u32::MAX"))?;
            self.special_to_id.insert(token.to_string(), id);
            self.id_to_special.push(token.to_string());
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.special_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        if id < BYTE_VOCAB_SIZE as u32 {
            None
        } else {
            let index = (id as usize) - BYTE_VOCAB_SIZE;
            self.id_to_special.get(index).map(|s| s.as_str())
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|b| *b as u32).collect()
    }

    pub fn encode_sequence<'a, I>(&self, sequence: I) -> Result<Vec<u32>>
    where
        I: IntoIterator<Item = EncodeInput<'a>>,
    {
        let mut encoded = Vec::new();
        for piece in sequence {
            match piece {
                EncodeInput::Text(text) => encoded.extend(self.encode(text)),
                EncodeInput::Special(name) => {
                    let id = self
                        .special_to_id
                        .get(name)
                        .copied()
                        .ok_or_else(|| anyhow!("Special token '{}' not registered", name))?;
                    encoded.push(id);
                }
            }
        }
        Ok(encoded)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut result = String::new();
        let mut byte_buffer: Vec<u8> = Vec::new();

        let flush_bytes = |buffer: &mut Vec<u8>, output: &mut String| -> Result<()> {
            if buffer.is_empty() {
                return Ok(());
            }
            let chunk = std::str::from_utf8(buffer)?;
            output.push_str(chunk);
            buffer.clear();
            Ok(())
        };

        for &token in tokens {
            if token < BYTE_VOCAB_SIZE as u32 {
                byte_buffer.push(token as u8);
            } else {
                flush_bytes(&mut byte_buffer, &mut result)?;
                let index = (token as usize) - BYTE_VOCAB_SIZE;
                let special = self
                    .id_to_special
                    .get(index)
                    .ok_or_else(|| anyhow!("Unknown special token id {}", token))?;
                result.push_str(special);
            }
        }

        flush_bytes(&mut byte_buffer, &mut result)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{BYTE_VOCAB_SIZE, EncodeInput, Tokenizer};

    #[test]
    fn byte_roundtrip() {
        let tokenizer = Tokenizer::new();
        let sentence = "Hello, GPU!";
        let tokens = tokenizer.encode(sentence);
        assert_eq!(tokens.len(), sentence.len());
        for (byte, &id) in sentence.as_bytes().iter().zip(tokens.iter()) {
            assert_eq!(*byte as u32, id);
        }
        let decoded = tokenizer.decode(&tokens).expect("decode");
        assert_eq!(decoded, sentence);
    }

    #[test]
    fn special_tokens_roundtrip() {
        let mut tokenizer = Tokenizer::with_special_tokens(["<bos>", "<eos>"]).expect("init");
        let sequence = [
            EncodeInput::Special("<bos>"),
            EncodeInput::Text("Hello"),
            EncodeInput::Text(" "),
            EncodeInput::Text("world"),
            EncodeInput::Special("<eos>"),
        ];
        let tokens = tokenizer.encode_sequence(sequence).expect("encode");
        assert_eq!(tokens.first().copied(), tokenizer.token_to_id("<bos>"));
        assert_eq!(tokens.last().copied(), tokenizer.token_to_id("<eos>"));

        assert_eq!(tokenizer.vocab_size(), BYTE_VOCAB_SIZE + 2);

        let decoded = tokenizer.decode(&tokens).expect("decode");
        assert_eq!(decoded, "<bos>Hello world<eos>");

        let ids = tokenizer
            .add_special_tokens(["<bos>", "<pad>"])
            .expect("add specials");
        assert_eq!(ids[0], tokenizer.token_to_id("<bos>").unwrap());
        assert_eq!(tokenizer.vocab_size(), BYTE_VOCAB_SIZE + 3);
        assert_eq!(tokenizer.id_to_token(ids[1]).unwrap(), "<pad>");
    }

    #[test]
    fn decode_rejects_unknown_special_id() {
        let tokenizer = Tokenizer::new();
        let err = tokenizer.decode(&[BYTE_VOCAB_SIZE as u32]).unwrap_err();
        assert!(err.to_string().contains("Unknown special token id"));
    }
}
