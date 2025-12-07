//! MIDAS Collector - High-performance Binance WebSocket data collector
//!
//! Collects L2 order book depth and trade streams from Binance Futures.
//! Stores raw messages in ZSTD-compressed JSONL format with automatic file rotation.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use zstd::stream::write::Encoder as ZstdEncoder;

/// Configuration loaded from environment variables
#[derive(Debug, Clone)]
struct Config {
    symbol: String,
    raw_data_path: PathBuf,
    file_rotation_hours: u64,
    file_rotation_bytes: u64,
    reconnect_delay_secs: u64,
    max_reconnect_attempts: u32, // 0 = infinite
}

impl Config {
    fn from_env() -> Result<Self> {
        Ok(Self {
            symbol: env::var("SYMBOL").unwrap_or_else(|_| "btcusdt".to_string()),
            raw_data_path: PathBuf::from(
                env::var("RAW_DATA_PATH").unwrap_or_else(|_| "/data/raw".to_string()),
            ),
            file_rotation_hours: env::var("FILE_ROTATION_HOURS")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .unwrap_or(1),
            file_rotation_bytes: env::var("FILE_ROTATION_GB")
                .unwrap_or_else(|_| "1".to_string())
                .parse::<u64>()
                .unwrap_or(1)
                * 1024
                * 1024
                * 1024,
            reconnect_delay_secs: env::var("RECONNECT_DELAY_SECS")
                .unwrap_or_else(|_| "5".to_string())
                .parse()
                .unwrap_or(5),
            max_reconnect_attempts: env::var("MAX_RECONNECT_ATTEMPTS")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
        })
    }
}

/// Raw depth update from Binance
#[derive(Debug, Deserialize)]
struct BinanceDepthUpdate {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "T")]
    transaction_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "U")]
    first_update_id: u64,
    #[serde(rename = "u")]
    last_update_id: u64,
    #[serde(rename = "pu")]
    prev_last_update_id: u64,
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,
}

/// Raw trade from Binance
#[derive(Debug, Deserialize)]
struct BinanceTrade {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "T")]
    trade_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "t")]
    trade_id: u64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "m")]
    buyer_is_maker: bool,
}

/// Combined stream message wrapper
#[derive(Debug, Deserialize)]
struct StreamMessage {
    stream: String,
    data: serde_json::Value,
}

/// Normalized output record for storage
#[derive(Debug, Serialize)]
struct OutputRecord {
    exchange_ts: i64,
    local_ts: i64,
    #[serde(rename = "type")]
    record_type: String,
    symbol: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_update_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_update_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prev_update_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bids: Option<Vec<[String; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asks: Option<Vec<[String; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trade: Option<TradeData>,
}

#[derive(Debug, Serialize)]
struct TradeData {
    trade_id: u64,
    price: String,
    quantity: String,
    buyer_is_maker: bool,
}

/// File writer with ZSTD compression and rotation
struct RotatingWriter {
    config: Config,
    current_file: Option<ZstdEncoder<'static, BufWriter<File>>>,
    current_path: PathBuf,
    file_start_time: Instant,
    bytes_written: u64,
    total_messages: Arc<AtomicU64>,
}

impl RotatingWriter {
    fn new(config: Config, total_messages: Arc<AtomicU64>) -> Result<Self> {
        fs::create_dir_all(&config.raw_data_path)?;
        let mut writer = Self {
            config,
            current_file: None,
            current_path: PathBuf::new(),
            file_start_time: Instant::now(),
            bytes_written: 0,
            total_messages,
        };
        writer.rotate_file()?;
        Ok(writer)
    }

    fn should_rotate(&self) -> bool {
        let time_exceeded =
            self.file_start_time.elapsed() >= Duration::from_secs(self.config.file_rotation_hours * 3600);
        let size_exceeded = self.bytes_written >= self.config.file_rotation_bytes;
        time_exceeded || size_exceeded
    }

    fn rotate_file(&mut self) -> Result<()> {
        // Finish and close current file
        if let Some(encoder) = self.current_file.take() {
            encoder.finish()?;
            info!(
                "Closed file {:?} ({} bytes written)",
                self.current_path, self.bytes_written
            );
        }

        // Create new file with timestamp
        let now: DateTime<Utc> = Utc::now();
        let filename = format!(
            "{}_{}.jsonl.zst",
            self.config.symbol,
            now.format("%Y%m%d_%H%M%S")
        );
        self.current_path = self.config.raw_data_path.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.current_path)
            .with_context(|| format!("Failed to create file: {:?}", self.current_path))?;

        let buffered = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
        let encoder = ZstdEncoder::new(buffered, 3)?; // Compression level 3

        self.current_file = Some(encoder);
        self.file_start_time = Instant::now();
        self.bytes_written = 0;

        info!("Created new data file: {:?}", self.current_path);
        Ok(())
    }

    fn write_record(&mut self, record: &OutputRecord) -> Result<()> {
        if self.should_rotate() {
            self.rotate_file()?;
        }

        let json = serde_json::to_string(record)?;
        if let Some(encoder) = &mut self.current_file {
            let line = format!("{}\n", json);
            let bytes = line.as_bytes();
            encoder.write_all(bytes)?;
            self.bytes_written += bytes.len() as u64;
            self.total_messages.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(encoder) = &mut self.current_file {
            encoder.flush()?;
        }
        Ok(())
    }
}

impl Drop for RotatingWriter {
    fn drop(&mut self) {
        if let Some(encoder) = self.current_file.take() {
            let _ = encoder.finish();
        }
    }
}

/// WebSocket collector
struct Collector {
    config: Config,
    running: Arc<AtomicBool>,
    total_messages: Arc<AtomicU64>,
}

impl Collector {
    fn new(config: Config) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(true)),
            total_messages: Arc::new(AtomicU64::new(0)),
        }
    }

    fn build_ws_url(&self) -> String {
        format!(
            "wss://fstream.binance.com/stream?streams={}@depth@100ms/{}@trade",
            self.config.symbol.to_lowercase(),
            self.config.symbol.to_lowercase()
        )
    }

    async fn connect(&self) -> Result<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>> {
        let url = self.build_ws_url();
        info!("Connecting to {}", url);

        let (ws_stream, response) = connect_async(&url)
            .await
            .with_context(|| format!("Failed to connect to {}", url))?;

        info!("Connected! Response: {:?}", response.status());
        Ok(ws_stream)
    }

    fn parse_message(&self, msg: &str) -> Result<OutputRecord> {
        let local_ts = Utc::now().timestamp_micros();
        let stream_msg: StreamMessage = serde_json::from_str(msg)?;

        if stream_msg.stream.contains("depth") {
            let depth: BinanceDepthUpdate = serde_json::from_value(stream_msg.data)?;
            Ok(OutputRecord {
                exchange_ts: depth.transaction_time * 1000, // Convert to micros
                local_ts,
                record_type: "depth".to_string(),
                symbol: depth.symbol,
                first_update_id: Some(depth.first_update_id),
                last_update_id: Some(depth.last_update_id),
                prev_update_id: Some(depth.prev_last_update_id),
                bids: Some(depth.bids),
                asks: Some(depth.asks),
                trade: None,
            })
        } else if stream_msg.stream.contains("trade") {
            let trade: BinanceTrade = serde_json::from_value(stream_msg.data)?;
            Ok(OutputRecord {
                exchange_ts: trade.trade_time * 1000, // Convert to micros
                local_ts,
                record_type: "trade".to_string(),
                symbol: trade.symbol,
                first_update_id: None,
                last_update_id: None,
                prev_update_id: None,
                bids: None,
                asks: None,
                trade: Some(TradeData {
                    trade_id: trade.trade_id,
                    price: trade.price,
                    quantity: trade.quantity,
                    buyer_is_maker: trade.buyer_is_maker,
                }),
            })
        } else {
            anyhow::bail!("Unknown stream type: {}", stream_msg.stream)
        }
    }

    async fn run(&self) -> Result<()> {
        let (tx, mut rx) = mpsc::channel::<OutputRecord>(100_000);
        let running = self.running.clone();
        let config = self.config.clone();
        let total_messages = self.total_messages.clone();

        // Writer task
        let writer_handle = tokio::spawn(async move {
            let mut writer = match RotatingWriter::new(config, total_messages) {
                Ok(w) => w,
                Err(e) => {
                    error!("Failed to create writer: {}", e);
                    return;
                }
            };

            let mut flush_interval = tokio::time::interval(Duration::from_secs(5));
            let mut last_count = 0u64;
            let mut stats_interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                tokio::select! {
                    Some(record) = rx.recv() => {
                        if let Err(e) = writer.write_record(&record) {
                            error!("Failed to write record: {}", e);
                        }
                    }
                    _ = flush_interval.tick() => {
                        if let Err(e) = writer.flush() {
                            error!("Failed to flush: {}", e);
                        }
                    }
                    _ = stats_interval.tick() => {
                        let current = writer.total_messages.load(Ordering::Relaxed);
                        let rate = (current - last_count) / 60;
                        info!(
                            "Stats: {} total messages, {} msg/sec, {} bytes in current file",
                            current, rate, writer.bytes_written
                        );
                        last_count = current;
                    }
                    else => break,
                }
            }

            info!("Writer task shutting down");
        });

        // Connection loop with reconnection logic
        let mut reconnect_attempts = 0u32;

        while running.load(Ordering::Relaxed) {
            match self.connect().await {
                Ok(ws_stream) => {
                    reconnect_attempts = 0;
                    info!("WebSocket connected, starting message loop");

                    let (mut write, mut read) = ws_stream.split();

                    // Ping task
                    let running_ping = running.clone();
                    let ping_handle = tokio::spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(30));
                        while running_ping.load(Ordering::Relaxed) {
                            interval.tick().await;
                            if let Err(e) = write.send(Message::Ping(vec![])).await {
                                warn!("Ping failed: {}", e);
                                break;
                            }
                        }
                    });

                    // Message loop
                    while let Some(msg_result) = read.next().await {
                        if !running.load(Ordering::Relaxed) {
                            break;
                        }

                        match msg_result {
                            Ok(Message::Text(text)) => {
                                match self.parse_message(&text) {
                                    Ok(record) => {
                                        if tx.send(record).await.is_err() {
                                            error!("Channel closed, exiting");
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        debug!("Failed to parse message: {} - {}", e, &text[..text.len().min(200)]);
                                    }
                                }
                            }
                            Ok(Message::Ping(data)) => {
                                debug!("Received ping");
                            }
                            Ok(Message::Pong(_)) => {
                                debug!("Received pong");
                            }
                            Ok(Message::Close(frame)) => {
                                warn!("Received close frame: {:?}", frame);
                                break;
                            }
                            Ok(Message::Binary(_)) => {
                                debug!("Received binary message (ignoring)");
                            }
                            Ok(Message::Frame(_)) => {}
                            Err(e) => {
                                error!("WebSocket error: {}", e);
                                break;
                            }
                        }
                    }

                    ping_handle.abort();
                    warn!("Disconnected from WebSocket");
                }
                Err(e) => {
                    error!("Connection failed: {}", e);
                }
            }

            // Reconnection logic
            reconnect_attempts += 1;
            if self.config.max_reconnect_attempts > 0
                && reconnect_attempts >= self.config.max_reconnect_attempts
            {
                error!(
                    "Max reconnection attempts ({}) reached, exiting",
                    self.config.max_reconnect_attempts
                );
                break;
            }

            if running.load(Ordering::Relaxed) {
                let delay = Duration::from_secs(self.config.reconnect_delay_secs);
                warn!(
                    "Reconnecting in {} seconds (attempt {})",
                    delay.as_secs(),
                    reconnect_attempts
                );
                sleep(delay).await;
            }
        }

        drop(tx);
        writer_handle.await?;
        Ok(())
    }

    fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("MIDAS Collector starting...");

    let config = Config::from_env()?;
    info!("Configuration: {:?}", config);

    let collector = Arc::new(Collector::new(config));
    let collector_clone = collector.clone();

    // Handle shutdown signals
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received shutdown signal");
        collector_clone.stop();
    });

    collector.run().await?;

    info!("MIDAS Collector stopped");
    Ok(())
}
