import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export interface MarketOverview {
  regime: string
  regime_description: string
  index_price: number
  index_change_pct: number
  index_high: number
  index_low: number
  index_open: number
  index_prev_close: number
  index_name: string
  sentiment_advance_ratio: number
  sentiment_up_count: number
  sentiment_down_count: number
  sentiment_limit_up: number
  sentiment_limit_down: number
  buy_threshold: number
  sell_threshold: number
  position_max: number
}

export interface KLineItem {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface StockResult {
  symbol: string
  name: string
  industry: string
  current_price: number
  change_pct: number
  probability: number
  composite_score: number
  reasons: string[]
  advice: string
  position_pct: number
  model_confidence: number
  rank: number
  predicted_return: number
  risk_level: string
}

export interface DailyAnalysis {
  date: string
  market_regime: string
  model_confidence: number
  advance_ratio: number
  recommended_position: number
  total_stocks: number
  results: StockResult[]
}

export interface SectorItem {
  name: string
  momentum_20d: number
  momentum_5d: number
  stock_count: number
  top_stocks: { name: string; composite_score: number }[]
}

export interface SectorAnalysis {
  sectors: SectorItem[]
  market_regime: string
}

export interface HoldingItem {
  symbol: string
  name?: string
  shares: number
  cost: number
}

export interface HoldingAnalysis {
  symbol: string
  name: string
  industry: string
  shares: number
  cost: number
  current_price: number
  market_value: number
  pnl: number
  pnl_pct: number
  change_today: number
  composite_score: number
  advice: string
  reasons: string[]
  rsi: number
  ret_5d: number
  ret_20d: number
}

export interface PortfolioAnalysis {
  holdings: HoldingAnalysis[]
  failed?: { code: string; reason: string }[]
  total_value: number
  total_cost: number
  total_pnl: number
  total_pnl_pct: number
  industry_allocation: Record<string, { value: number; pct: number; pnl: number; stocks: string[] }>
  regime: string
  buy_count: number
  sell_count: number
  loss_count: number
}

export async function getMarketOverview(): Promise<MarketOverview> {
  const r = await api.get('/market/overview')
  return r.data
}

export async function getMarketHistory(days = 60): Promise<{ data: KLineItem[] }> {
  const r = await api.get('/market/history', { params: { days } })
  return r.data
}

export async function getDailyAnalysis(): Promise<DailyAnalysis> {
  const r = await api.get('/analysis/daily')
  return r.data
}

export interface AnalysisReport {
  text: string
  error?: string
}

export async function getAnalysisReport(): Promise<AnalysisReport> {
  const r = await api.get('/analysis/report')
  return r.data
}

export async function sendSlackReport(): Promise<{ message: string }> {
  const r = await api.post('/analysis/send-slack')
  return r.data
}

export async function refreshAnalysis(): Promise<{ message: string }> {
  const r = await api.post('/analysis/refresh')
  return r.data
}

export interface SlackConfig {
  token: string
  channel: string
}

export async function getSlackConfig(): Promise<SlackConfig> {
  const r = await api.get('/settings/slack')
  return r.data
}

export async function saveSlackConfig(cfg: SlackConfig): Promise<{ message: string }> {
  const r = await api.put('/settings/slack', cfg)
  return r.data
}

export interface AnalysisStatus {
  status: string
  progress: number
  message: string
}

export async function getAnalysisStatus(): Promise<AnalysisStatus> {
  const r = await api.get('/analysis/status')
  return r.data
}

export async function getSectors(): Promise<SectorAnalysis> {
  const r = await api.get('/sectors')
  return r.data
}

export async function getPortfolio(): Promise<{ holdings: HoldingItem[] }> {
  const r = await api.get('/portfolio')
  return r.data
}

export async function savePortfolio(holdings: HoldingItem[]): Promise<{ message: string; count: number }> {
  const r = await api.put('/portfolio', { holdings })
  return r.data
}

export async function analyzePortfolio(holdings: HoldingItem[]): Promise<PortfolioAnalysis> {
  const r = await api.post('/portfolio/analyze', { holdings })
  return r.data
}
