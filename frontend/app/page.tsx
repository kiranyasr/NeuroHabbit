"use client"
import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts'
import { createClient } from '@supabase/supabase-js'

/* ================= ⚙️ CONFIGURATION ================= */
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
const supabase = createClient(supabaseUrl, supabaseKey)

/* ================= 🌊 GLOW-BEAT VECTOR WAVEFORM ================= */
const GlowBeatWave = ({ color }: { color: string }) => (
  <div className="absolute bottom-0 left-0 w-full h-16 overflow-hidden opacity-30 pointer-events-none">
    <svg className="absolute bottom-0 w-[200%] h-full animate-flow-beat" viewBox="0 0 1000 100" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`glow-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.8" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path 
        d="M0,40 C150,90 350,10 500,40 C650,90 850,10 1000,40 L1000,100 L0,100 Z" 
        fill={`url(#glow-${color})`} 
      />
    </svg>
    <style jsx>{`
      @keyframes flow-beat {
        0% { transform: translateX(0) scaleY(1); }
        50% { transform: translateX(-25%) scaleY(1.3); } 
        100% { transform: translateX(-50%) scaleY(1); }
      }
      .animate-flow-beat {
        animation: flow-beat 3s ease-in-out infinite;
      }
    `}</style>
  </div>
)

/* ================= 📊 CHART COMPONENT ================= */
const RiskChart = ({ data }: { data: any[] }) => (
  <div className="bg-[#f3f4f6] rounded-2xl border border-[#1a2d42]/10 p-8 shadow-sm h-[320px] flex-1">
    <h4 className="text-[10px] font-black text-[#9ca3af] uppercase tracking-widest mb-6">Neural Frequency Pulse</h4>
    <ResponsiveContainer width="100%" height="80%">
      <BarChart data={data}>
        <CartesianGrid vertical={false} stroke="#d1d5db" strokeOpacity={0.5} />
        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#1a2d42', fontSize: 10, fontWeight: 700 }} />
        <YAxis hide />
        <Tooltip cursor={{ fill: '#d1d5db' }} contentStyle={{ border: 'none', borderRadius: '8px', background: '#1a2d42', color: '#fffdf6' }} />
        <Bar dataKey="risk" barSize={30} radius={[4, 4, 0, 0]}>
          {data.map((_, i) => <Cell key={i} fill={i % 2 === 0 ? '#1a2d42' : '#9ca3af'} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>
)

/* ================= 🚀 MAIN SYSTEM ================= */
export default function HabitFlowFullSystem() {
  const [session, setSession] = useState<any>(null)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [trend, setTrend] = useState<any>(null)
  const [streakData, setStreakData] = useState<any>(null)
  const [heatmap, setHeatmap] = useState<any[]>([])
  const [nps, setNps] = useState<any>(null)
  const [report, setReport] = useState("")
  const [copyStatus, setCopyStatus] = useState(false)
  const [analyticsData, setAnalyticsData] = useState<any[]>([])

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session))
    const { data: listener } = supabase.auth.onAuthStateChange((_e, s) => setSession(s))
    return () => listener.subscription.unsubscribe()
  }, [])

  const headers = { "Content-Type": "application/json", "Authorization": `Bearer ${session?.access_token}` }

  const fetchData = async () => {
    if (!session) return
    const [h, s, hm, n] = await Promise.all([
      fetch("http://localhost:8000/get-habits", { headers }).then(r => r.json()),
      fetch("http://localhost:8000/habit-streak", { headers }).then(r => r.json()),
      fetch("http://localhost:8000/activity-heatmap", { headers }).then(r => r.json()),
      fetch("http://localhost:8000/neural-performance-score", { headers }).then(r => r.json())
    ])
    setAnalyticsData((h.habits || []).slice(0, 5).map((x: any) => ({ name: x.activity || "UNK", risk: x.risk_score || 0 })))
    setStreakData(s); setHeatmap(hm); setNps(n)
  }

  useEffect(() => { if (session) fetchData() }, [session])

  const handleAnalyze = async () => {
    if (!text || !session) return
    setLoading(true)
    const res = await fetch("http://localhost:8000/add-habit", { method: "POST", headers, body: JSON.stringify({ text }) })
    const data = await res.json()
    setResult(data)
    fetchData(); setText(""); setLoading(false)
  }

  if (!session) {
    return (
      <div className="min-h-screen bg-[#d1d5db] flex items-center justify-center p-6">
        <div className="bg-[#f3f4f6] p-12 rounded-2xl border border-[#1a2d42]/10 shadow-xl w-full max-w-md">
          <div className="flex justify-center mb-8">
             <div className="w-12 h-12 bg-[#1a2d42] rounded-lg rotate-3 shadow-lg" />
          </div>
          <input className="w-full p-4 mb-4 rounded-xl bg-white border border-[#1a2d42]/10 outline-none focus:ring-2 focus:ring-[#9ca3af]" placeholder="IDENTIFIER" onChange={e => setEmail(e.target.value)} />
          <input type="password" title="password" className="w-full p-4 mb-8 rounded-xl bg-white border border-[#1a2d42]/10 outline-none focus:ring-2 focus:ring-[#9ca3af]" placeholder="ACCESS_KEY" onChange={e => setPassword(e.target.value)} />
          <button onClick={() => supabase.auth.signInWithPassword({ email, password })} className="w-full py-4 bg-[#1a2d42] text-[#fffdf6] font-black uppercase text-xs tracking-widest rounded-xl hover:bg-[#9ca3af] transition-all">Initialize</button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen w-screen bg-[#d1d5db] font-sans text-[#1a2d42] overflow-x-hidden pb-20">
      
      {/* 🧭 NAVIGATION */}
      <nav className="h-[80px] bg-[#f3f4f6]/80 backdrop-blur-md px-12 flex items-center justify-between border-b border-[#1a2d42]/10 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <motion.div 
            whileHover={{ rotate: 90 }}
            className="w-10 h-10 bg-[#1a2d42] rounded-lg flex items-center justify-center relative overflow-hidden shadow-md"
          >
            <div className="absolute inset-0 bg-white opacity-10" />
            <div className="w-4 h-4 border-2 border-[#fffdf6] rounded-sm rotate-45" />
          </motion.div>
          <h1 className="font-black text-2xl tracking-tighter uppercase">Habit<span className="text-[#9ca3af]">Flow</span></h1>
        </div>
        <button 
          onClick={() => supabase.auth.signOut()} 
          className="text-[10px] font-black uppercase tracking-widest px-4 py-2 rounded-lg bg-[#4b5563] text-[#f8fafc] border border-[#1a2d42]/20 hover:bg-red-600 transition-all shadow-sm"
        >
          Terminate Session
        </button>
      </nav>

      <main className="max-w-[1400px] mx-auto p-8 space-y-8">
        
      {/* 📊 METRIC BOXES: Light Grey with Rounded Corners */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { label: 'System Efficiency', val: `${nps?.nps_score || 0}%`, color: '#9333ea' },
            { label: 'Biological Momentum', val: `${streakData?.streak || 0}d`, color: '#2563eb' },
            { label: 'Risk Probability', val: result?.prediction || "0.00%", color: '#db2777' },
            { label: 'Trend Delta', val: trend?.trend_vector || "Neutral", color: '#059669' }
          ].map((m, i) => (
            <div 
              key={i}
              className="bg-slate-200 p-8 rounded-2xl border border-slate-300 relative overflow-hidden group shadow-sm transition-transform hover:-translate-y-1"
            >
              <div className="relative z-10">
                <p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500 mb-2">{m.label}</p>
                <h2 className="text-4xl font-black tracking-tighter" style={{ color: m.color }}>{m.val}</h2>
              </div>
              <GlowBeatWave color={m.color} />
            </div>
          ))}
        </div>

        {/* ⌨️ COMMAND CENTER */}
        <div className="bg-[#f3f4f6] p-4 rounded-2xl border border-[#1a2d42]/10 flex gap-4 shadow-sm">
          <input 
            className="flex-1 px-6 py-4 rounded-xl bg-white text-[#1a2d42] font-bold outline-none border border-transparent focus:border-[#d1d5db] transition-all"
            value={text} onChange={e => setText(e.target.value)} placeholder="Sync behavior telemetry..."
          />
          <button onClick={handleAnalyze} className="px-12 rounded-xl bg-[#1a2d42] text-[#fffdf6] font-black uppercase text-xs tracking-widest hover:bg-[#9ca3af] transition-all">Execute</button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <RiskChart data={analyticsData} />
          </div>

          <div className="bg-[#f3f4f6] p-8 rounded-2xl border border-[#1a2d42]/10 flex flex-col justify-between h-[320px] shadow-sm">
             <div>
                <h4 className="text-[10px] font-black text-[#9ca3af] uppercase tracking-widest mb-6">Activity Heatmap</h4>
                <div className="grid grid-cols-8 gap-2">
                  {heatmap.length ? heatmap.map((h, i) => (
                    <div 
                      key={i} 
                      className="h-8 rounded-md" 
                      style={{ backgroundColor: h.count ? `rgba(26, 45, 66, ${Math.min(0.2 + h.count * 0.4, 1)})` : '#d1d5db' }} 
                    />
                  )) : Array(24).fill(0).map((_, i) => <div key={i} className="h-8 bg-[#d1d5db]/30 rounded-md" />)}
                </div>
             </div>
             <button onClick={() => {
                setLoading(true)
                fetch("http://localhost:8000/generate-report", { headers }).then(r => r.json()).then(d => { setReport(d.report); setLoading(false) })
              }} className="w-full py-4 rounded-xl bg-[#d1d5db] text-[#1a2d42] font-black text-[10px] uppercase tracking-widest hover:bg-[#1a2d42] hover:text-white transition-all">
                Synthesize Report
              </button>
          </div>
        </div>

        {/* 📜 REPORT OUTPUT */}
        <AnimatePresence>
          {report && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }} 
              animate={{ opacity: 1, scale: 1 }} 
              className="bg-white p-12 rounded-3xl border border-[#1a2d42] shadow-2xl relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-1 h-full bg-[#1a2d42]" />
              <p className="text-[10px] font-black uppercase text-[#9ca3af] mb-4 tracking-[0.3em]">Synthesis Result</p>
              <h3 className="text-3xl font-black italic text-[#1a2d42] mb-8 leading-tight">"{report}"</h3>
              <div className="flex gap-6">
                <button onClick={() => { navigator.clipboard.writeText(report); setCopyStatus(true); setTimeout(() => setCopyStatus(false), 2000) }} className="text-xs font-black uppercase border-b-2 border-[#1a2d42] pb-1">
                  {copyStatus ? "Copied" : "Capture"}
                </button>
                <button onClick={() => setReport("")} className="text-xs font-black uppercase text-[#9ca3af]">Dismiss</button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  )
}