use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

const DATASET_ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";
const DATASET: &str = "open-phi/textbooks";
const CONFIG: &str = "default";
const SPLIT: &str = "train";
const BATCH_SIZE: usize = 100;
const REQUEST_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Deserialize)]
struct RowsResponse {
    #[serde(default)]
    num_rows_total: usize,
    rows: Vec<RowWrapper>,
}

#[derive(Debug, Deserialize)]
struct RowWrapper {
    row: RowData,
}

#[derive(Debug, Deserialize)]
struct RowData {
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    concepts: Option<String>,
    #[serde(default)]
    outline: Option<String>,
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    field: Option<String>,
    #[serde(default)]
    subfield: Option<String>,
    #[serde(default)]
    rag: Option<String>,
}

/// Download the `open-phi/textbooks` dataset using the Hugging Face datasets-server REST API
/// and emit a newline-delimited UTF-8 text file.
pub fn download_textbooks<P: AsRef<Path>>(output_path: P) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .context("Failed to construct HTTP client")?;

    let output_path = output_path.as_ref();
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file at {:?}", output_path))?;
    let mut writer = BufWriter::new(file);

    let mut offset = 0usize;
    let mut total_rows = None;
    let mut rows_written = 0usize;

    loop {
        let response = client
            .get(DATASET_ENDPOINT)
            .query(&[
                ("dataset", DATASET),
                ("config", CONFIG),
                ("split", SPLIT),
                ("offset", &offset.to_string()),
                ("length", &BATCH_SIZE.to_string()),
            ])
            .send()
            .with_context(|| format!("HTTP request failed at offset {}", offset))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Dataset server returned {} at offset {}",
                response.status(),
                offset
            ));
        }

        let payload: RowsResponse = response
            .json()
            .with_context(|| format!("Failed to decode JSON response at offset {}", offset))?;

        if let Some(expected_total) = total_rows {
            if payload.num_rows_total != 0 && payload.num_rows_total != expected_total {
                return Err(anyhow!(
                    "Dataset total row count changed during download (was {}, now {})",
                    expected_total,
                    payload.num_rows_total
                ));
            }
        } else if payload.num_rows_total == 0 {
            return Err(anyhow!(
                "Dataset server reported zero rows for {}/{}:{}",
                DATASET,
                CONFIG,
                SPLIT
            ));
        } else {
            total_rows = Some(payload.num_rows_total);
        }

        if payload.rows.is_empty() {
            break;
        }

        for row in payload.rows {
            let combined = combine_row_text(&row.row);
            if combined.is_empty() {
                continue;
            }
            writer
                .write_all(combined.as_bytes())
                .context("Failed to write text row")?;
            writer.write_all(b"\n").context("Failed to write newline")?;
            rows_written += 1;
        }

        offset += BATCH_SIZE;

        if let Some(expected_total) = total_rows {
            if offset >= expected_total {
                break;
            }
        }
    }

    writer.flush().context("Failed to flush output writer")?;

    if rows_written == 0 {
        return Err(anyhow!(
            "No textual rows were written to {:?}; dataset may be empty",
            output_path
        ));
    }

    Ok(())
}

fn combine_row_text(row: &RowData) -> String {
    let mut sections: Vec<String> = Vec::new();

    if let Some(topic) = row
        .topic
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sections.push(format!("# {}", topic));
    }

    if let Some(outline) = row
        .outline
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sections.push(outline.to_string());
    }

    if let Some(markdown) = row
        .markdown
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sections.push(markdown.to_string());
    }

    if let Some(model) = row
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sections.push(format!("Model: {}", model));
    }

    if let Some(concepts) = row
        .concepts
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "[]")
    {
        sections.push(format!("Concepts: {}", concepts));
    }

    if let Some(field) = row
        .field
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sections.push(format!("Field: {}", field));
    }

    if let Some(subfield) = row
        .subfield
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sections.push(format!("Subfield: {}", subfield));
    }

    if let Some(rag) = row.rag.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        sections.push(format!("Source: {}", rag));
    }

    if sections.is_empty() {
        return String::new();
    }

    sections.join("\n\n").replace('\r', " ").replace('\n', " ")
}
