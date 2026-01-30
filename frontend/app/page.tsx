"use client"
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// 📈 ANALYTICS COMPONENT
const RiskChart = ({ data }: { data: any[] }) => (
  <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm h-64">
    <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-4">Behavioral Friction Analysis</h4>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
        <XAxis 
          dataKey="name" 
          fontSize={10} 
          tick={{fill: '#64748b'}} 
          axisLine={false} 
          tickLine={false}
          style={{ fontFamily: '"Times New Roman", Times, serif' }}
        />
        <YAxis 
          fontSize={10} 
          tick={{fill: '#64748b'}} 
          axisLine={false} 
          tickLine={false}
          style={{ fontFamily: '"Times New Roman", Times, serif' }}
        />
        <Tooltip 
          cursor={{fill: '#f8fafc'}} 
          contentStyle={{fontFamily: 'Times New Roman', fontSize: '12px', borderRadius: '8px'}} 
        />
        <Bar dataKey="risk" fill="#E30613" radius={[4, 4, 0, 0]} barSize={35} />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

export default function NeuralManifestDashboard() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [trend, setTrend] = useState<any>(null)
  const [streakData, setStreakData] = useState<any>(null)
  const [heatmap, setHeatmap] = useState<any[]>([])
  const [metrics, setMetrics] = useState({ status: "LOADING", latency: "0ms" });
  const [nudge, setNudge] = useState<{style: string, message: string} | null>(null);
  const [report, setReport] = useState(""); 
  const [copyStatus, setCopyStatus] = useState(false);
  const [nps, setNps] = useState<any>(null); // 🔥 Day 20: Performance Score State
  const [history, setHistory] = useState<any[]>([])
  const [analyticsData, setAnalyticsData] = useState<any[]>([])

  // --- 🔔 UTILITIES ---
  const triggerProactiveAlert = (hour: number) => {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("NEUROHABIT: PRE-EMPTIVE STRIKE", {
        body: `Behavioral patterns suggest a peak window at ${hour}:00. Initiate sequence.`,
        tag: "proactive-alert"
      });
    }
  };

  const handleGenerateReport = async () => {
    try {
      setLoading(true);
      const res = await fetch("http://localhost:8000/generate-report");
      const data = await res.json();
      setReport(data.report);
      if (data.report) {
        await navigator.clipboard.writeText(data.report);
        setCopyStatus(true);
        setTimeout(() => setCopyStatus(false), 3000);
      }
    } catch (e) { console.error("Identity Sync Failed"); } finally { setLoading(false); }
  };

  const fetchData = async () => {
    try {
      const res = await fetch("http://localhost:8000/get-habits")
      if (!res.ok) throw new Error("Offline");
      const data = await res.json()
      setHistory(data.habits || [])
      setAnalyticsData((data.habits || []).slice(0, 5).map((h: any) => ({
        name: h.activity || "UNKNOWN",
        risk: h.risk_score || 0 
      })))
    } catch (error) { console.error(error) }
  }

  useEffect(() => {
    const saved = localStorage.getItem('lastNudge');
    if (saved) setNudge(JSON.parse(saved));
    if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();

    const runSystemSync = () => {
      fetchData();
      fetch("http://localhost:8000/habit-streak").then(res => res.json()).then(data => setStreakData(data)).catch(() => null);
      fetch("http://localhost:8000/activity-heatmap").then(res => res.json()).then(data => setHeatmap(data)).catch(() => null);
      fetch("http://localhost:8000/system-health").then(res => res.json()).then(data => setMetrics(data)).catch(() => null);
      fetch("http://localhost:8000/neural-performance-score").then(res => res.json()).then(data => setNps(data)).catch(() => null);
      fetch("http://localhost:8000/next-prediction").then(res => res.json()).then(data => {
            const currentHourUTC = new Date().getUTCHours();
            if (data?.next_window === currentHourUTC + 1) triggerProactiveAlert(data.next_window);
        }).catch(() => null);
    };

    runSystemSync();
    const interval = setInterval(runSystemSync, 30000); 
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!text) return
    setLoading(true)
    try {
      const res = await fetch("http://localhost:8000/add-habit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.toUpperCase() })
      })
      const data = await res.json()
      
      // 🚀 Fix: Encode URI Component to handle spaces like "MORINGA WATER"
      const trendRes = await fetch(`http://localhost:8000/habit-trends/${encodeURIComponent(data.saved_data.activity)}`);
      const trendData = await trendRes.json();
      
      setTrend(trendData);
      setResult(data);
      setNudge(data.nudge);
      localStorage.setItem('lastNudge', JSON.stringify(data.nudge));
      
      fetchData();
      setText("");
    } catch (err) { console.error(err) } finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#1A2238] selection:bg-[#E30613]/20" 
         style={{ fontFamily: '"Times New Roman", Times, serif' }}>
      
      <nav className="bg-[#1A2238] text-white px-8 py-4 flex justify-between items-center sticky top-0 z-50 shadow-md">
        <div className="flex items-center gap-3">
          <div className="bg-[#E30613] w-8 h-8 flex items-center justify-center rounded-sm">
             <span className="text-xs font-bold text-white">NH</span>
          </div>
          <h1 className="text-lg font-bold tracking-widest uppercase">NEURO<span className="text-[#E30613]">HABIT</span></h1>
        </div>
        <div className="flex gap-6 text-[10px] font-bold uppercase tracking-widest text-slate-400">
          <div className="flex items-center gap-1.5">
            <motion.div animate={{ opacity: [1, 0.3, 1], scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 2 }}
              className="w-2 h-2 rounded-full bg-[#E30613] shadow-[0_0_8px_#E30613]" />
            <span className="text-white border-b-2 border-[#E30613]">Live Tracker</span>
          </div>
          <span className="hover:text-white transition-colors cursor-pointer uppercase">Logs</span>
          <span className="hover:text-white transition-colors cursor-pointer uppercase">Stats</span>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-10 space-y-6">

        {/* 🔥 Day 20: Performance Score HUD */}
        {nps && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden">
            <div className="absolute left-0 top-0 h-1 bg-[#E30613]" style={{ width: `${nps.nps_score}%` }} />
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Neural Performance Score</h2>
                <p className="text-2xl font-black text-[#1A2238]">{nps.nps_score}<span className="text-xs text-slate-400">/100</span></p>
              </div>
              <div className="text-right">
                <span className={`text-[9px] font-bold px-3 py-1 rounded-full uppercase ${nps.nps_score > 75 ? 'bg-emerald-500 text-white' : 'bg-[#1A2238] text-white'}`}>
                  Status: {nps.rating}
                </span>
              </div>
            </div>
          </motion.div>
        )}
        
        <AnimatePresence>
          {nudge && nudge.style && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
              className="bg-[#1A2238] border-l-4 border-[#E30613] p-6 rounded-r-xl shadow-lg relative overflow-hidden">
              <div className="absolute top-0 right-0 p-2 opacity-10">
                <span className="text-4xl text-white font-bold">{nudge.style[0]}</span>
              </div>
              <div className="flex justify-between items-start">
                <h5 className="text-[10px] font-bold text-[#E30613] uppercase tracking-[0.3em] mb-2">AI Protocol: {nudge.style} Nudge</h5>
                <button onClick={() => { localStorage.removeItem('lastNudge'); setNudge(null); }}
                  className="text-[9px] text-slate-500 hover:text-[#E30613] uppercase font-bold transition-colors">Dismiss</button>
              </div>
              <p className="text-white text-sm font-medium leading-relaxed">"{nudge.message}"</p>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="bg-[#1A2238] p-8 rounded-tr-[2rem] rounded-bl-[2rem] shadow-xl relative overflow-hidden">
           <div className="relative z-10 space-y-4">
              <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.3em]">Initialize Habit Sequence</h3>
              <div className="flex flex-col md:flex-row gap-3">
                <input className="flex-1 bg-white border border-slate-200 rounded-lg px-5 py-3 text-base font-bold text-[#1A2238] outline-none uppercase shadow-inner" 
                  placeholder="E.G. GYM FOR 45 MINUTES" value={text} onChange={(e) => setText(e.target.value.toUpperCase())} />
                <button onClick={handleAnalyze} className="bg-[#E30613] text-white px-8 py-3 rounded-lg text-xs font-bold uppercase tracking-widest hover:bg-[#C10510] transition-all">
                  {loading ? "PARSING..." : "DEPLOY HABIT"}
                </button>
              </div>
           </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-sm">
            <p className="text-[10px] font-bold text-slate-400 uppercase mb-1 tracking-widest">Risk Prediction</p>
            <p className="text-3xl font-bold text-[#E30613]">{result ? result.prediction : "--%"}</p>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-sm">
            <p className="text-[10px] font-bold text-slate-400 uppercase mb-1 tracking-widest">Momentum Vector</p>
            <p className={`text-xl font-bold ${trend?.trend_vector === 'UPWARD' ? 'text-emerald-600' : trend?.trend_vector === 'DECLINING' ? 'text-[#E30613]' : 'text-slate-600'}`}>
              {trend?.trend_vector ? trend.trend_vector.replace('_', ' ') : "STABLE"}
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-sm flex flex-col items-center justify-center">
            <p className="text-[10px] font-bold text-slate-400 uppercase mb-1 tracking-widest">Neural Streak</p>
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-bold text-[#1A2238]">{streakData?.streak || 0}</span>
              <span className="text-[10px] font-bold text-slate-400 uppercase">Days</span>
            </div>
            {streakData && <span className="mt-2 text-[8px] font-bold bg-[#F1F5F9] text-[#E30613] px-2 py-0.5 rounded-full uppercase">{streakData.level}</span>}
          </div>
        </div>

        <div className="bg-[#1A2238] p-6 rounded-2xl shadow-xl border border-[#E30613]/30">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="text-[10px] font-bold text-[#E30613] uppercase tracking-[0.2em]">Neural Identity Sync</h4>
              <p className="text-slate-400 text-[10px] mt-1 uppercase tracking-tight">Synthesize behavioral footprint for clipboard export.</p>
            </div>
            <button onClick={handleGenerateReport} className="bg-white text-[#1A2238] px-4 py-2 rounded-lg text-[9px] font-bold uppercase hover:bg-[#E30613] hover:text-white transition-all shadow-lg">
              {loading ? "PROFILING..." : "Generate Report"}
            </button>
          </div>
          <AnimatePresence>
            {(report || copyStatus) && (
              <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="mt-4 p-4 bg-black/30 rounded border border-white/5">
                {report && <p className="text-[11px] text-slate-300 font-mono italic leading-relaxed">"{report}"</p>}
                {copyStatus && <div className="mt-2 text-[8px] text-emerald-500 uppercase font-bold tracking-widest text-right">Report Copied to Clipboard ✓</div>}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
          <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-4">Neuro-Activity Heatmap (24H Cycle)</h4>
          <div className="grid grid-cols-6 md:grid-cols-12 gap-2">
            {heatmap.length > 0 ? heatmap.map((slot, i) => (
              <motion.div key={i} title={`Hour ${slot.hour}:00 | ${slot.count} Events`} className="h-8 rounded-sm transition-all"
                style={{ backgroundColor: slot.count > 0 ? `rgba(227, 6, 19, ${Math.min(0.2 + (slot.count * 0.25), 1)})` : '#f1f5f9', border: slot.count > 4 ? '1px solid #E30613' : 'none' }} />
            )) : Array(24).fill(0).map((_, i) => <div key={i} className="h-8 bg-slate-50 animate-pulse rounded-sm" />)}
          </div>
          <div className="flex justify-between mt-2 text-[8px] text-slate-400 font-bold uppercase tracking-tighter">
            <span>00:00 Night</span><span>12:00 PM</span><span>23:00 Night</span>
          </div>
        </div>

        <RiskChart data={analyticsData} />

        <section className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-6 border-t border-slate-200">
           <div className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm text-center">
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Neural Accuracy</h4>
             <p className="text-xl font-bold text-[#1A2238]">99.8% <span className="text-[10px] text-[#E30613]">SYNC</span></p>
           </div>
           <div className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm text-center">
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">System Load</h4>
             <p className="text-xl font-bold text-[#1A2238]">{metrics.latency} <span className="text-[10px] text-[#E30613]">LATENCY</span></p>
           </div>
           <div className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm text-center">
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Active Monitors</h4>
             <p className="text-xl font-bold text-[#1A2238]">{metrics.status} <span className="text-[10px] text-[#E30613]">LIVE</span></p>
           </div>
        </section>
      </main>
    </div>
  )
}