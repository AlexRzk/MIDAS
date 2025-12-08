//! MIDAS Collector v2.0 - High-performance Binance WebSocket data collector
//!
//! Collects L2 order book depth, trade, liquidation, and kline streams from Binance Futures.
//! Features:
//! - Sequence validation with internal order book state
//! - Bounded mpsc channel for backpressure
//! - Buffered writes to reduce syscalls
//! - Rich metadata (connection info, dropped message counters)
//! - Optional stream types (trades, liquidations, klines)

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use zstd::stream::write::Encoder as ZstdEncoder;

// ============================================
// Configuration
// ============================================

#[derive(Debug, Clone)]
struct Config {
    symbol: String,
    raw_data_path: PathBuf,
    file_rotation_hours: u64,
    file_rotation_bytes: u64,
    reconnect_delay_secs: u64,
    max_reconnect_attempts: u32,
    // Optional streams
    enable_trades: bool,
    enable_liquidations: bool,
    enable_klines: bool,
    kline_interval: String,
    // Channel buffer size for backpressure
    channel_buffer_size: usize,
    // Write buffer size
    write_buffer_size: usize,
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
                * 1024 * 1024 * 1024,
            reconnect_delay_secs: env::var("RECONNECT_DELAY_SECS")
                .unwrap_or_else(|_| "5".to_string())
                .parse()
                .unwrap_or(5),
            max_reconnect_attempts: env::var("MAX_RECONNECT_ATTEMPTS")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
            enable_trades: env::var("ENABLE_TRADES")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            enable_liquidations: env::var("ENABLE_LIQUIDATIONS")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            enable_klines: env::var("ENABLE_KLINES")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            kline_interval: env::var("KLINE_INTERVAL")
                .unwrap_or_else(|_| "1m".to_string()),
            channel_buffer_size: env::var("CHANNEL_BUFFER_SIZE")
                .unwrap_or_else(|_| "100000".to_string())
                .parse()
                .unwrap_or(100_000),
            write_buffer_size: env::var("WRITE_BUFFER_SIZE")
                .unwrap_or_else(|_| "2097152".to_string())
                .parse()
                .unwrap_or(2 * 1024 * 1024),
        })
    }
}

// ============================================
// Binance Message Types
// ============================================

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

#[derive(Debug, Deserialize)]
struct BinanceLiquidation {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "o")]
    order: LiquidationOrder,
}

#[derive(Debug, Deserialize)]
struct LiquidationOrder {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "S")]
    side: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "ap")]
    avg_price: String,
    #[serde(rename = "X")]
    status: String,
    #[serde(rename = "T")]
    trade_time: i64,
}

#[derive(Debug, Deserialize)]
struct BinanceKline {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    kline: KlineData,
}

#[derive(Debug, Deserialize)]
struct KlineData {
    #[serde(rename = "i")]
    interval: String,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "n")]
    trades: u64,
    #[serde(rename = "x")]
    is_final: bool,
}

#[derive(Debug, Deserialize)]
struct StreamMessage {
    stream: String,
    data: serde_json::Value,
}

// ============================================
// Output Records
// ============================================

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
    #[serde(skip_serializing_if = "Option::is_none")]
    liquidation: Option<LiquidationData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kline: Option<KlineOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta: Option<RecordMetadata>,
}

#[derive(Debug, Serialize)]
struct TradeData {
    trade_id: u64,
    price: String,
    quantity: String,
    buyer_is_maker: bool,
}

#[derive(Debug, Serialize)]
struct LiquidationData {
    side: String,
    price: String,
    quantity: String,
    avg_price: String,
    status: String,
}

#[derive(Debug, Serialize)]
struct KlineOutput {
    interval: String,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: String,
    trades: u64,
    is_final: bool,
}

#[derive(Debug, Serialize, Clone)]
struct RecordMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    sequence_gap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_prev_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    book_crossed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_bid: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_ask: Option<f64>,
}

// ============================================
// Internal Order Book State (for validation)
// ============================================

struct InternalOrderBook {
    bids: BTreeMap<u64, f64>,
    asks: BTreeMap<u64, f64>,
    last_update_id: u64,
    initialized: bool,
}

impl InternalOrderBook {
    fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            initialized: false,
        }
    }

    fn price_to_key(price: &str) -> u64 {
        let p: f64 = price.parse().unwrap_or(0.0);
        (p * 100_000_000.0) as u64
    }

    fn apply_update(&mut self, depth: &BinanceDepthUpdate) -> (bool, Option<RecordMetadata>) {
        let mut meta = RecordMetadata {
            sequence_gap: None,
            expected_prev_id: None,
            book_crossed: None,
            best_bid: None,
            best_ask: None,
        };

        let has_gap = if self.initialized {
            if depth.prev_last_update_id != self.last_update_id {
                meta.sequence_gap = Some(true);
                meta.expected_prev_id = Some(self.last_update_id);
                true
            } else {
                false
            }
        } else {
            self.initialized = true;
            false
        };

        for [price_str, size_str] in &depth.bids {
            let key = Self::price_to_key(price_str);
            let size: f64 = size_str.parse().unwrap_or(0.0);
            if size == 0.0 {
                self.bids.remove(&key);
            } else {
                self.bids.insert(key, size);
            }
        }

        for [price_str, size_str] in &depth.asks {
            let key = Self::price_to_key(price_str);
            let size: f64 = size_str.parse().unwrap_or(0.0);
            if size == 0.0 {
                self.asks.remove(&key);
            } else {
                self.asks.insert(key, size);
            }
        }

        self.last_update_id = depth.last_update_id;

        let best_bid = self.bids.keys().next_back().map(|&k| k as f64 / 100_000_000.0);
        let best_ask = self.asks.keys().next().map(|&k| k as f64 / 100_000_000.0);

        meta.best_bid = best_bid;
        meta.best_ask = best_ask;

        if let (Some(bb), Some(ba)) = (best_bid, best_ask) {
            if bb >= ba {
                meta.book_crossed = Some(true);
            }
        }

        let has_issues = has_gap || meta.book_crossed.unwrap_or(false);
        (has_issues, if has_issues || best_bid.is_some() { Some(meta) } else { None })
    }

    fn reset(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_update_id = 0;
        self.initialized = false;
    }
}

// ============================================
// Collector Statistics
// ============================================

#[derive(Debug, Default)]
struct CollectorStats {
    total_messages: AtomicU64,
    depth_messages: AtomicU64,
    trade_messages: AtomicU64,
    liquidation_messages: AtomicU64,
    kline_messages: AtomicU64,
    sequence_gaps: AtomicU64,
    crossed_books: AtomicU64,
    dropped_messages: AtomicU64,
    parse_errors: AtomicU64,
    reconnections: AtomicU64,
}

impl CollectorStats {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

// ============================================
// Rotating Writer with Buffering
// ============================================

struct RotatingWriter {
    config: Config,
    current_file: Option<ZstdEncoder<'static, BufWriter<File>>>,
    current_path: PathBuf,
    file_start_time: Instant,
    bytes_written: u64,
    stats: Arc<CollectorStats>,
    connection_id: u64,
}

impl RotatingWriter {
    fn new(config: Config, stats: Arc<CollectorStats>) -> Result<Self> {
        debug!("Creating RotatingWriter with path: {:?}", config.raw_data_path);
        fs::create_dir_all(&config.raw_data_path)
            .with_context(|| format!("Failed to create directory: {:?}", config.raw_data_path))?;
        
        // Test write permissions
        let test_file = config.raw_data_path.join(".writetest");
        fs::write(&test_file, b"test")
            .with_context(|| format!("No write permission in {:?}", config.raw_data_path))?;
        fs::remove_file(&test_file).ok();
        
        let mut writer = Self {
            config,
            current_file: None,
            current_path: PathBuf::new(),
            file_start_time: Instant::now(),
            bytes_written: 0,
            stats,
            connection_id: 0,
        };
        writer.rotate_file()?;
        debug!("RotatingWriter created successfully");
        Ok(writer)
    }

    fn should_rotate(&self) -> bool {
        let time_exceeded =
            self.file_start_time.elapsed() >= Duration::from_secs(self.config.file_rotation_hours * 3600);
        let size_exceeded = self.bytes_written >= self.config.file_rotation_bytes;
        time_exceeded || size_exceeded
    }

    fn rotate_file(&mut self) -> Result<()> {
        if let Some(mut encoder) = self.current_file.take() {
            let footer = self.create_file_footer();
            let footer_json = serde_json::to_string(&footer)?;
            let footer_line = format!("{{\"_footer\":{}}}\n", footer_json);
            encoder.write_all(footer_line.as_bytes())?;
            encoder.finish()?;
            info!(
                "Closed file {:?} ({} bytes written)",
                self.current_path, self.bytes_written
            );
        }

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

        let buffered = BufWriter::with_capacity(self.config.write_buffer_size, file);
        let encoder = ZstdEncoder::new(buffered, 3)?;

        self.current_file = Some(encoder);
        self.file_start_time = Instant::now();
        self.bytes_written = 0;

        let header = self.create_file_header();
        self.write_raw(&serde_json::to_string(&header)?)?;

        info!("Created new data file: {:?}", self.current_path);
        Ok(())
    }

    fn create_file_header(&self) -> serde_json::Value {
        serde_json::json!({
            "_header": {
                "version": "2.0",
                "symbol": self.config.symbol,
                "created_at": Utc::now().to_rfc3339(),
                "connection_id": self.connection_id,
                "streams": {
                    "depth": true,
                    "trades": self.config.enable_trades,
                    "liquidations": self.config.enable_liquidations,
                    "klines": self.config.enable_klines,
                }
            }
        })
    }

    fn create_file_footer(&self) -> serde_json::Value {
        serde_json::json!({
            "closed_at": Utc::now().to_rfc3339(),
            "bytes_written": self.bytes_written,
            "stats": {
                "total_messages": self.stats.total_messages.load(Ordering::Relaxed),
                "depth_messages": self.stats.depth_messages.load(Ordering::Relaxed),
                "trade_messages": self.stats.trade_messages.load(Ordering::Relaxed),
                "sequence_gaps": self.stats.sequence_gaps.load(Ordering::Relaxed),
                "crossed_books": self.stats.crossed_books.load(Ordering::Relaxed),
                "dropped_messages": self.stats.dropped_messages.load(Ordering::Relaxed),
            }
        })
    }

    fn write_record(&mut self, record: &OutputRecord) -> Result<()> {
        if self.should_rotate() {
            self.rotate_file()?;
        }

        let json = serde_json::to_string(record)?;
        self.write_raw(&json)?;
        self.stats.total_messages.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn write_raw(&mut self, json: &str) -> Result<()> {
        if let Some(encoder) = &mut self.current_file {
            let line = format!("{}\n", json);
            let bytes = line.as_bytes();
            encoder.write_all(bytes)?;
            self.bytes_written += bytes.len() as u64;
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

// ============================================
// Collector
// ============================================

struct Collector {
    config: Config,
    running: Arc<AtomicBool>,
    stats: Arc<CollectorStats>,
}

impl Collector {
    fn new(config: Config) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(true)),
            stats: CollectorStats::new(),
        }
    }

    fn build_ws_url(&self) -> String {
        let mut streams = vec![format!("{}@depth@100ms", self.config.symbol.to_lowercase())];

        if self.config.enable_trades {
            streams.push(format!("{}@trade", self.config.symbol.to_lowercase()));
        }

        if self.config.enable_liquidations {
            streams.push(format!("{}@forceOrder", self.config.symbol.to_lowercase()));
        }

        if self.config.enable_klines {
            streams.push(format!(
                "{}@kline_{}",
                self.config.symbol.to_lowercase(),
                self.config.kline_interval
            ));
        }

        format!(
            "wss://fstream.binance.com/stream?streams={}",
            streams.join("/")
        )
    }

    async fn connect(&self) -> Result<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>> {
        let url = self.build_ws_url();
        info!("Connecting to {}", url);

        let (ws_stream, response) = connect_async(&url)
            .await
            .map_err(|e| {
                error!("WebSocket connection error details: {:?}", e);
                e
            })
            .with_context(|| format!("Failed to connect to {}", url))?;

        info!("Connected! Response: {:?}", response.status());
        Ok(ws_stream)
    }

    fn parse_message(&self, msg: &str, order_book: &mut InternalOrderBook) -> Result<OutputRecord> {
        let local_ts = Utc::now().timestamp_micros();
        let stream_msg: StreamMessage = serde_json::from_str(msg)?;

        if stream_msg.stream.contains("depth") {
            let depth: BinanceDepthUpdate = serde_json::from_value(stream_msg.data)?;
            let (has_issues, meta) = order_book.apply_update(&depth);

            if has_issues {
                if meta.as_ref().map(|m| m.sequence_gap.unwrap_or(false)).unwrap_or(false) {
                    self.stats.sequence_gaps.fetch_add(1, Ordering::Relaxed);
                }
                if meta.as_ref().map(|m| m.book_crossed.unwrap_or(false)).unwrap_or(false) {
                    self.stats.crossed_books.fetch_add(1, Ordering::Relaxed);
                }
            }

            self.stats.depth_messages.fetch_add(1, Ordering::Relaxed);

            Ok(OutputRecord {
                exchange_ts: depth.transaction_time * 1000,
                local_ts,
                record_type: "depth".to_string(),
                symbol: depth.symbol,
                first_update_id: Some(depth.first_update_id),
                last_update_id: Some(depth.last_update_id),
                prev_update_id: Some(depth.prev_last_update_id),
                bids: Some(depth.bids),
                asks: Some(depth.asks),
                trade: None,
                liquidation: None,
                kline: None,
                meta,
            })
        } else if stream_msg.stream.contains("trade") {
            let trade: BinanceTrade = serde_json::from_value(stream_msg.data)?;
            self.stats.trade_messages.fetch_add(1, Ordering::Relaxed);

            Ok(OutputRecord {
                exchange_ts: trade.trade_time * 1000,
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
                liquidation: None,
                kline: None,
                meta: None,
            })
        } else if stream_msg.stream.contains("forceOrder") {
            let liq: BinanceLiquidation = serde_json::from_value(stream_msg.data)?;
            self.stats.liquidation_messages.fetch_add(1, Ordering::Relaxed);

            Ok(OutputRecord {
                exchange_ts: liq.order.trade_time * 1000,
                local_ts,
                record_type: "liquidation".to_string(),
                symbol: liq.order.symbol,
                first_update_id: None,
                last_update_id: None,
                prev_update_id: None,
                bids: None,
                asks: None,
                trade: None,
                liquidation: Some(LiquidationData {
                    side: liq.order.side,
                    price: liq.order.price,
                    quantity: liq.order.quantity,
                    avg_price: liq.order.avg_price,
                    status: liq.order.status,
                }),
                kline: None,
                meta: None,
            })
        } else if stream_msg.stream.contains("kline") {
            let kline: BinanceKline = serde_json::from_value(stream_msg.data)?;
            self.stats.kline_messages.fetch_add(1, Ordering::Relaxed);

            Ok(OutputRecord {
                exchange_ts: kline.event_time * 1000,
                local_ts,
                record_type: "kline".to_string(),
                symbol: kline.symbol,
                first_update_id: None,
                last_update_id: None,
                prev_update_id: None,
                bids: None,
                asks: None,
                trade: None,
                liquidation: None,
                kline: Some(KlineOutput {
                    interval: kline.kline.interval,
                    open: kline.kline.open,
                    high: kline.kline.high,
                    low: kline.kline.low,
                    close: kline.kline.close,
                    volume: kline.kline.volume,
                    trades: kline.kline.trades,
                    is_final: kline.kline.is_final,
                }),
                meta: None,
            })
        } else {
            anyhow::bail!("Unknown stream type: {}", stream_msg.stream)
        }
    }

    async fn run(&self) -> Result<()> {
        let (tx, mut rx) = mpsc::channel::<OutputRecord>(self.config.channel_buffer_size);
        let running = self.running.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();

        // Writer task
        let writer_handle = tokio::spawn(async move {
            debug!("Writer task starting...");
            let mut writer = match RotatingWriter::new(config, stats.clone()) {
                Ok(w) => {
                    info!("Writer initialized successfully");
                    w
                },
                Err(e) => {
                    error!("Failed to create writer: {}", e);
                    return;
                }
            };

            let mut flush_interval = tokio::time::interval(Duration::from_secs(5));
            let mut last_count = 0u64;
            let mut stats_interval = tokio::time::interval(Duration::from_secs(60));

            debug!("Writer task entering main loop");
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
                        let current = stats.total_messages.load(Ordering::Relaxed);
                        let rate = (current - last_count) / 60;
                        let gaps = stats.sequence_gaps.load(Ordering::Relaxed);
                        let dropped = stats.dropped_messages.load(Ordering::Relaxed);
                        debug!(
                            "Stats: {} total, {} msg/sec, {} gaps, {} dropped, {} bytes",
                            current, rate, gaps, dropped, writer.bytes_written
                        );
                        last_count = current;
                    }
                    else => {
                        debug!("Writer task: channel closed, exiting loop");
                        break;
                    }
                }
            }

            info!("Writer task shutting down");
        });

        // Connection loop
        let mut reconnect_attempts = 0u32;
        let mut connection_id = 0u64;

        while running.load(Ordering::Relaxed) {
            match self.connect().await {
                Ok(ws_stream) => {
                    reconnect_attempts = 0;
                    connection_id += 1;
                    info!("WebSocket connected (connection #{})", connection_id);

                    let (mut write, mut read) = ws_stream.split();
                    let mut order_book = InternalOrderBook::new();

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
                                match self.parse_message(&text, &mut order_book) {
                                    Ok(record) => {
                                        match tx.try_send(record) {
                                            Ok(_) => {}
                                            Err(mpsc::error::TrySendError::Full(_record)) => {
                                                self.stats.dropped_messages.fetch_add(1, Ordering::Relaxed);
                                                warn!("Channel full, dropping message (backpressure)");
                                            }
                                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                                error!("Channel closed unexpectedly - writer task likely crashed");
                                                break;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        self.stats.parse_errors.fetch_add(1, Ordering::Relaxed);
                                        debug!("Parse error: {} - {}", e, &text[..text.len().min(200)]);
                                    }
                                }
                            }
                            Ok(Message::Ping(_)) => {
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
                    self.stats.reconnections.fetch_add(1, Ordering::Relaxed);
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("MIDAS Collector v2.0 starting...");

    let config = Config::from_env()?;
    info!("Configuration: {:?}", config);

    let collector = Arc::new(Collector::new(config));
    let collector_clone = collector.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received shutdown signal");
        collector_clone.stop();
    });

    collector.run().await?;

    let stats = &collector.stats;
    info!(
        "Final stats: {} total, {} depth, {} trades, {} gaps, {} dropped",
        stats.total_messages.load(Ordering::Relaxed),
        stats.depth_messages.load(Ordering::Relaxed),
        stats.trade_messages.load(Ordering::Relaxed),
        stats.sequence_gaps.load(Ordering::Relaxed),
        stats.dropped_messages.load(Ordering::Relaxed),
    );

    info!("MIDAS Collector stopped");
    Ok(())
}
