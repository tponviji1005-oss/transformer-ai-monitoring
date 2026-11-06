import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Flame,
  Gauge,
  HelpCircle,
  Info,
  RefreshCw,
  Server,
  Settings2,
  ShieldAlert,
  WifiOff,
  Zap,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import { api } from './api';

const TRANSFORMERS = [
  { id: 'T1', name: 'T1 - Academic Block', spec: '315 kVA | 1-Phase', focus: 'Isolation Forest Anomaly Detection' },
  { id: 'T2', name: 'T2 - Hostel Block', spec: '315 kVA | 1-Phase', focus: 'Supervised Overload Classifier (33.82% Thresh)' },
  { id: 'T3', name: 'T3 - Facilities Block', spec: '250 kVA | 1-Phase', focus: 'Threshold Monitor (80% / 100%)' },
];

export default function App() {
  const [selectedId, setSelectedId] = useState('T1');
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [features, setFeatures] = useState(null);
  const [hourlyHistory, setHourlyHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [error, setError] = useState(null);

  // Nameplate modal state
  const [showSpecModal, setShowSpecModal] = useState(false);
  const [specForm, setSpecForm] = useState({
    rated_kVA: 315.0,
    phases: 1,
    rated_pf: 0.9,
    verified: true,
  });

  const fetchData = useCallback(async (tId) => {
    try {
      setError(null);
      const [hRes, pRes, rRes, fRes, histRes] = await Promise.allSettled([
        api.getHealth(),
        api.getLatestPrediction(tId),
        api.getFeatureReadiness(tId),
        api.getLatestFeatures(tId),
        api.getHourlyHistory(tId, 24),
      ]);

      if (hRes.status === 'fulfilled') {
        setHealth(hRes.value);
      } else {
        setHealth(null);
      }

      if (pRes.status === 'fulfilled') setPrediction(pRes.value);
      else setPrediction(null);

      if (rRes.status === 'fulfilled') setReadiness(rRes.value);
      else setReadiness(null);

      if (fRes.status === 'fulfilled') setFeatures(fRes.value);
      else setFeatures(null);

      if (histRes.status === 'fulfilled') {
        const sorted = (histRes.value.records || []).map((r) => ({
          time: new Date(r.hour_start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          power_kW: r.power_W != null ? Number((r.power_W / 1000).toFixed(1)) : null,
          oil_temp_C: r.oil_temperature_C != null ? Number(r.oil_temperature_C.toFixed(1)) : null,
          ambient_temp_C: r.ambient_temperature_C != null ? Number(r.ambient_temperature_C.toFixed(1)) : null,
          voltage_V: r.voltage_V != null ? Number(r.voltage_V.toFixed(1)) : null,
          current_A: r.current_A != null ? Number(r.current_A.toFixed(1)) : null,
          frequency_Hz: r.frequency_Hz != null ? Number(r.frequency_Hz.toFixed(2)) : null,
          power_factor: r.power_factor != null ? Number(r.power_factor.toFixed(2)) : null,
          reading_count: r.reading_count,
        })).reverse();
        setHourlyHistory(sorted);
      } else {
        setHourlyHistory([]);
      }
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData(selectedId);
  }, [selectedId, fetchData]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchData(selectedId);
    }, 10000);
    return () => clearInterval(interval);
  }, [autoRefresh, selectedId, fetchData]);

  const handleUpdateSpecs = async (e) => {
    e.preventDefault();
    try {
      await api.updateSpecs(selectedId, {
        rated_kVA: parseFloat(specForm.rated_kVA),
        phases: parseInt(specForm.phases, 10),
        rated_pf: parseFloat(specForm.rated_pf),
        verified: Boolean(specForm.verified),
      });
      setShowSpecModal(false);
      fetchData(selectedId);
    } catch (err) {
      alert(`Failed to update nameplate: ${err.message}`);
    }
  };

  const getAlertBadge = (level) => {
    switch (level) {
      case 'CRITICAL':
        return { bg: 'bg-rose-500/20 text-rose-300 border-rose-500/40', icon: ShieldAlert, border: 'border-rose-500' };
      case 'WARNING':
        return { bg: 'bg-amber-500/20 text-amber-300 border-amber-500/40', icon: AlertTriangle, border: 'border-amber-500' };
      case 'NORMAL':
        return { bg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', icon: CheckCircle2, border: 'border-emerald-500' };
      default:
        return { bg: 'bg-slate-800/80 text-slate-300 border-slate-700', icon: Info, border: 'border-slate-700' };
    }
  };

  const alertStyle = getAlertBadge(prediction?.alert_level);
  const AlertIcon = alertStyle.icon;

  // Latest verified telemetry observation
  const latestTelemetry = hourlyHistory.length > 0 ? hourlyHistory[hourlyHistory.length - 1] : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-indigo-500/10 border border-indigo-500/30 rounded-xl text-indigo-400">
              <Zap className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                Transformer AI Monitoring
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 font-mono">
                  v1.0.0
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Predictive Maintenance & Multi-Sensor Telemetry System (I-BLEND Benchmarked)
              </p>
            </div>
          </div>

          {/* System Status Controls */}
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
              <span className={`w-2 h-2 rounded-full ${health?.database_connected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-slate-300">
                Backend: <strong className={health?.database_connected ? 'text-emerald-300' : 'text-rose-400'}>
                  {health?.database_connected ? 'Connected' : 'Offline / Degraded'}
                </strong>
              </span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-400">Raw Readings: {health?.total_raw_readings ?? '--'}</span>
            </div>

            <button
              onClick={() => setShowSpecModal(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 transition"
              title="Configure verified transformer nameplate specifications"
            >
              <Settings2 className="w-3.5 h-3.5" />
              Nameplate Specs
            </button>

            <button
              onClick={() => fetchData(selectedId)}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-xs text-white font-medium transition disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 py-6 flex-1 w-full space-y-6">
        {/* Offline / Backend Error Notification */}
        {(!health || !health.database_connected) && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2.5">
              <WifiOff className="w-4 h-4 flex-shrink-0" />
              <span>
                <strong>Backend Service Unreachable:</strong> Telemetry API is not responding at <code className="bg-rose-950/50 px-1 py-0.5 rounded">/api/health</code>. Showing cached telemetry data.
              </span>
            </div>
            <button
              onClick={() => fetchData(selectedId)}
              className="px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 font-medium transition text-white"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Transformer Unit Switcher Tabs */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {TRANSFORMERS.map((t) => {
            const isSelected = selectedId === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setSelectedId(t.id)}
                className={`p-4 rounded-xl text-left border transition-all ${
                  isSelected
                    ? 'bg-slate-800/90 border-indigo-500/80 shadow-lg shadow-indigo-500/10'
                    : 'bg-slate-900/50 border-slate-800 hover:bg-slate-800/50 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">{t.name}</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-slate-300">
                    {t.id}
                  </span>
                </div>
                <div className="text-xs text-slate-400 mt-1 flex items-center gap-1">
                  <span>Capacity: {t.spec}</span>
                </div>
                <div className="text-[11px] text-indigo-400/90 mt-2 font-mono">
                  {t.focus}
                </div>
              </button>
            );
          })}
        </div>

        {/* Unified Operator Advisory Alert Banner */}
        <div className={`p-5 rounded-2xl border ${alertStyle.border} ${alertStyle.bg} transition-all`}>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              <AlertIcon className="w-6 h-6 mt-0.5 flex-shrink-0" />
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h2 className="text-base font-bold tracking-tight">
                    {prediction?.alert_type || (prediction?.status === 'UNAVAILABLE' ? 'INFERENCE GATED / UNAVAILABLE' : 'NO ACTIVE PREDICTION')}
                  </h2>
                  <span className="text-xs px-2.5 py-0.5 rounded-full font-mono uppercase font-bold border border-current">
                    {prediction?.alert_level || 'AWAITING TELEMETRY'}
                  </span>
                </div>
                <p className="text-xs opacity-90 mt-1 leading-relaxed">
                  {prediction?.alert_reason || prediction?.status_reason || 'Accumulating telemetry stream. Next-hour predictive inference activates upon completed hourly windows.'}
                </p>
                {prediction?.recommended_action && (
                  <div className="mt-3 p-3 rounded-lg bg-black/30 border border-white/10 text-xs">
                    <strong className="block text-white mb-1">Recommended Operator Action:</strong>
                    <span>{prediction.recommended_action}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="text-right text-xs opacity-75 hidden sm:block whitespace-nowrap font-mono">
              Observation: {prediction?.feature_hour_timestamp ? new Date(prediction.feature_hour_timestamp).toLocaleString() : 'N/A'}
            </div>
          </div>
        </div>

        {/* Real-Time Telemetry & Forecast KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Active Power */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
              <span>Active Power (kW)</span>
              <Zap className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold text-white font-mono">
              {latestTelemetry?.power_kW != null
                ? `${latestTelemetry.power_kW.toFixed(1)} kW`
                : (features?.power_W != null ? `${(features.power_W / 1000).toFixed(1)} kW` : '--')}
            </div>
            <div className="text-xs text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-2">
              <span>Next-Hour Forecast:</span>
              <strong className="text-indigo-400 font-mono">
                {prediction?.predicted_power_W != null
                  ? `${(prediction.predicted_power_W / 1000).toFixed(1)} kW`
                  : 'Unavailable'}
              </strong>
            </div>
          </div>

          {/* Load Percentage */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
              <span>Transformer Load</span>
              <Gauge className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="text-2xl font-bold text-white font-mono">
              {features?.load_pct != null
                ? `${features.load_pct.toFixed(1)}%`
                : (readiness?.readiness_status === 'STATUS_BLOCKED_MISSING_NAMEPLATE' ? (
                  <span className="text-xs text-amber-400 font-normal">Unverified Nameplate</span>
                ) : '--')}
            </div>
            <div className="text-xs text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-2">
              <span>Predicted Load:</span>
              <strong className={`font-mono ${prediction?.predicted_load_pct >= 80 ? 'text-rose-400' : 'text-indigo-400'}`}>
                {prediction?.predicted_load_pct != null ? `${prediction.predicted_load_pct.toFixed(1)}%` : 'Unavailable'}
              </strong>
            </div>
          </div>

          {/* Top-Oil Temperature */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
              <span>Top-Oil Temp (MAX6675)</span>
              <Flame className="w-4 h-4 text-rose-400" />
            </div>
            <div className="text-2xl font-bold text-white font-mono">
              {latestTelemetry?.oil_temp_C != null
                ? `${latestTelemetry.oil_temp_C.toFixed(1)} °C`
                : (features?.oil_temp_C != null ? `${features.oil_temp_C.toFixed(1)} °C` : '--')}
            </div>
            <div className="text-xs text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-2">
              <span>Thermal Bounds:</span>
              <span className="text-slate-400 font-mono">Warn: 85°C | Crit: 105°C</span>
            </div>
          </div>

          {/* Ambient Temperature (Strict Null Enforcement) */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
              <span>Ambient Temp</span>
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-sm font-mono mt-1">
              {latestTelemetry?.ambient_temp_C != null ? (
                <span className="text-2xl font-bold text-white">{latestTelemetry.ambient_temp_C.toFixed(1)} °C</span>
              ) : (
                <div className="space-y-1">
                  <span className="inline-block px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono text-xs">
                    NULL (Unconfigured)
                  </span>
                  <p className="text-[10px] text-slate-400 leading-tight">
                    No physical ambient sensor wired. Never substituted with oil temperature.
                  </p>
                </div>
              )}
            </div>
            <div className="text-xs text-slate-500 mt-2 border-t border-slate-800/80 pt-1.5">
              Affects 6 historical lag features
            </div>
          </div>
        </div>

        {/* Unit-Specific Machine Learning Panels */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* ML Health & Anomaly Panel */}
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              Machine Learning Diagnostic Engine
            </h3>

            {/* T2 Overload Panel */}
            {selectedId === 'T2' && (
              <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/80 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">T2 Overload Classifier Probability</span>
                  <span className="font-mono font-bold text-indigo-400">
                    {prediction?.overload_probability != null
                      ? `${(prediction.overload_probability * 100).toFixed(1)}%`
                      : 'Unavailable'}
                  </span>
                </div>
                <div className="w-full bg-slate-700 h-2.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      (prediction?.overload_probability ?? 0) >= 0.3382 ? 'bg-rose-500' : 'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.min(100, (prediction?.overload_probability ?? 0) * 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                  <span>Decision Threshold: 33.82%</span>
                  <span>Alert: {prediction?.is_overload_predicted ? 'OVERLOAD PREDICTED' : 'Nominal'}</span>
                </div>
              </div>
            )}

            {/* T1 Isolation Forest Panel */}
            {selectedId === 'T1' && (
              <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/80 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">T1 Isolation Forest Anomaly Score</span>
                  <span className="font-mono font-bold text-indigo-400">
                    {prediction?.anomaly_score != null ? prediction.anomaly_score.toFixed(4) : 'Unavailable'}
                  </span>
                </div>
                <div className="text-xs text-slate-400">
                  Anomaly Flag:{' '}
                  <strong className={prediction?.is_anomaly_detected ? 'text-rose-400' : 'text-emerald-400'}>
                    {prediction?.is_anomaly_detected ? 'OUT-OF-DISTRIBUTION DETECTED' : 'Nominal Pattern'}
                  </strong>
                </div>
              </div>
            )}

            {/* T3 Threshold Monitor Panel */}
            {selectedId === 'T3' && (
              <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/80 space-y-2 text-xs">
                <span className="text-slate-300 font-medium">T3 Threshold Monitor</span>
                <p className="text-slate-400 leading-relaxed">
                  Zero historical overloads in training set. Evaluates active capacity thresholds (80% Warning / 100% Critical).
                </p>
                <div className="flex gap-4 text-xs font-mono pt-1 text-slate-300">
                  <span>Current Load: {features?.load_pct != null ? `${features.load_pct.toFixed(1)}%` : '--'}</span>
                  <span>Warning: 80%</span>
                  <span>Critical: 100%</span>
                </div>
              </div>
            )}

            {/* 39-Feature Vector Diagnostics */}
            <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 space-y-1.5 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-300">39-Feature Vector Status:</span>
                <span className="font-mono text-indigo-400 font-medium">
                  {readiness?.readiness_status || 'CHECKING PREREQUISITES...'}
                </span>
              </div>
              <div className="flex justify-between items-center text-slate-400">
                <span>Inference Readiness:</span>
                <span className={readiness?.is_ready_for_inference ? 'text-emerald-400 font-medium' : 'text-amber-400'}>
                  {readiness?.is_ready_for_inference ? 'READY (All 39 Valid)' : 'GATED (Prerequisites Incomplete)'}
                </span>
              </div>
              {readiness?.missing_features?.length > 0 && (
                <div className="text-[11px] text-amber-300/80 mt-1">
                  Missing ({readiness.missing_features.length}): {readiness.missing_features.slice(0, 3).join(', ')}
                  {readiness.missing_features.length > 3 ? ` (+${readiness.missing_features.length - 3} more)` : ''}
                </div>
              )}
            </div>
          </div>

          {/* Physical Hardware & Transmission Channel Telemetry */}
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Server className="w-4 h-4 text-emerald-400" />
              Verified Hardware Measurements (PZEM-004T v3.0)
            </h3>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-slate-400 block mb-1">Measured Voltage</span>
                <span className="text-lg font-bold text-white font-mono">
                  {latestTelemetry?.voltage_V != null ? `${latestTelemetry.voltage_V.toFixed(1)} V` : '--'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-slate-400 block mb-1">Measured Current</span>
                <span className="text-lg font-bold text-white font-mono">
                  {latestTelemetry?.current_A != null ? `${latestTelemetry.current_A.toFixed(1)} A` : '--'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-slate-400 block mb-1">Grid Frequency</span>
                <span className="text-lg font-bold text-white font-mono">
                  {latestTelemetry?.frequency_Hz != null ? `${latestTelemetry.frequency_Hz.toFixed(2)} Hz` : '--'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-slate-400 block mb-1">Power Factor</span>
                <span className="text-lg font-bold text-white font-mono">
                  {latestTelemetry?.power_factor != null ? latestTelemetry.power_factor.toFixed(2) : '--'}
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/40 text-[11px] text-slate-400 space-y-1">
              <div><strong>UART2 Link:</strong> GPIO 16 (RX) / GPIO 17 (TX) &bull; 9600 baud</div>
              <div><strong>SPI Thermocouple:</strong> GPIO 19 (SCK) / GPIO 23 (SO) / GPIO 5 (CS)</div>
              <div><strong>Firmware Protection:</strong> Monotonic boot sequence ID &bull; 60-reading RAM ring buffer</div>
            </div>
          </div>
        </div>

        {/* 24-Hour Telemetry & Trend Visualizations */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Clock className="w-4 h-4 text-indigo-400" />
              24-Hour Verified Active Power & Thermal Trend
            </h3>
            <span className="text-xs text-slate-400 font-mono">Last Updated: {lastRefreshed || 'N/A'}</span>
          </div>

          <div className="h-64 w-full">
            {hourlyHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hourlyHistory}>
                  <defs>
                    <linearGradient id="powerGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} unit=" kW" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Area type="monotone" dataKey="power_kW" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#powerGrad)" name="Active Power (kW)" />
                  <Line type="monotone" dataKey="oil_temp_C" stroke="#f43f5e" strokeWidth={2} dot={false} name="Top-Oil Temp (°C)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-xs text-slate-500 gap-2 border border-dashed border-slate-800 rounded-xl">
                <HelpCircle className="w-8 h-8 text-slate-600" />
                <span>No historical hourly records accumulated yet for {selectedId}.</span>
                <span className="text-[11px] text-slate-600">Charts automatically plot as ESP32 2-second readings complete their hourly aggregation windows.</span>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Nameplate Configuration Modal */}
      {showSpecModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex justify-between items-center">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-indigo-400" />
                Configure Nameplate Rating: {selectedId}
              </h3>
              <button onClick={() => setShowSpecModal(false)} className="text-slate-400 hover:text-white text-lg">&times;</button>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Establishing verified nameplate capacity is mandatory for load percentage calculation (<code className="text-indigo-300 font-mono">P / (S_rated * pf)</code>). Unverified units strictly gate live predictions to prevent erroneous overload alarms.
            </p>

            <form onSubmit={handleUpdateSpecs} className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-300 mb-1">Rated Capacity (kVA)</label>
                <input
                  type="number"
                  step="0.1"
                  value={specForm.rated_kVA}
                  onChange={(e) => setSpecForm({ ...specForm, rated_kVA: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white font-mono focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-300 mb-1">Phase Configuration</label>
                <select
                  value={specForm.phases}
                  onChange={(e) => setSpecForm({ ...specForm, phases: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value={1}>1-Phase (Single physical CT PZEM-004T sensor)</option>
                  <option value={3}>3-Phase Balanced System</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-300 mb-1">Rated Power Factor (pf)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.1"
                  max="1.0"
                  value={specForm.rated_pf}
                  onChange={(e) => setSpecForm({ ...specForm, rated_pf: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white font-mono focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>

              <div className="flex items-center gap-2 pt-2">
                <input
                  type="checkbox"
                  id="verifiedCheck"
                  checked={specForm.verified}
                  onChange={(e) => setSpecForm({ ...specForm, verified: e.target.checked })}
                  className="rounded border-slate-700 text-indigo-600 focus:ring-0"
                />
                <label htmlFor="verifiedCheck" className="text-slate-300 select-none">
                  Mark specification as officially verified from on-site nameplate
                </label>
              </div>

              <div className="flex justify-end gap-2 pt-4">
                <button
                  type="button"
                  onClick={() => setShowSpecModal(false)}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium"
                >
                  Save Specification
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-800/80 px-6 py-4 text-center text-xs text-slate-500">
        Transformer AI Predictive Maintenance System &bull; Madurai Hardware Deployment &bull; Milestone 5 Stage 5
      </footer>
    </div>
  );
}
