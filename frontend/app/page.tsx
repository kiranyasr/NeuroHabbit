"use client"
import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { createClient } from '@supabase/supabase-js'
import { 
  Weight, Zap, Activity, Clock, LogOut, ShieldCheck, 
  BrainCircuit, X, TrendingUp, Calendar, Loader2, ChevronDown, ChevronUp
} from 'lucide-react'
import { ResponsiveContainer, AreaChart, Area, YAxis, CartesianGrid, Tooltip } from 'recharts'

// --- 🛰️ INITIALIZATION ---
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
const supabase = createClient(supabaseUrl, supabaseKey)
const API_BASE = "http://localhost:8000" 

/* ================= 📊 VISUAL ANIMATIONS ================= */

const OrganicWave = ({ color }: { color: string }) => (
  <div className="absolute bottom-0 left-0 w-full h-16 opacity-20 pointer-events-none">
    <svg className="w-[200%] h-full animate-wave-fast" viewBox="0 0 1000 100" preserveAspectRatio="none">
      <path d="M0,60 C150,110 350,10 500,60 C650,110 850,10 1000,60 L1000,100 L0,100 Z" fill={color} />
    </svg>
    <style jsx>{`
      @keyframes wave-fast { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
      .animate-wave-fast { animation: wave-fast 4s linear infinite; }
    `}</style>
  </div>
)

const BarPulse = ({ color }: { color: string }) => (
  <div className="flex items-end gap-1 h-8">
    {[0.4, 0.7, 1, 0.6, 0.8].map((op, i) => (
      <motion.div
        key={i}
        animate={{ height: ["20%", "100%", "20%"] }}
        transition={{ duration: 1, repeat: Infinity, delay: i * 0.1 }}
        className="w-1 rounded-full"
        style={{ backgroundColor: color, opacity: op }}
      />
    ))}
  </div>
)

/* ================= 🚀 NEURO AI MAIN UI ================= */

export default function NeuroAI_Dashboard() {
  const [session, setSession] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [isModalOpen, setIsModalOpen] = useState(false)
  
  // State for metrics
  const [currentWeight, setCurrentWeight] = useState<number>(0)
  const [baselineWeight, setBaselineWeight] = useState<number>(0)
  const [inputBaseline, setInputBaseline] = useState("")
  const [inputCurrent, setInputCurrent] = useState("")
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      if (data.session) fetchWeights(data.session.access_token)
      else setLoading(false)
    })
    return () => clearInterval(timer)
  }, [])

  const fetchWeights = async (token: string) => {
    try {
      const res = await fetch(`${API_BASE}/get-latest-weight`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      const data = await res.json()
      const cw = data?.current_weight ?? 0.0
      const bw = data?.initial_weight ?? 0.0
      setCurrentWeight(cw)
      setBaselineWeight(bw)
      setInputCurrent(cw.toString())
      setInputBaseline(bw.toString())
    } catch (e) { console.error("Sync Failure:", e) } finally { setLoading(false) }
  }

  const handleUpdate = async () => {
    if (!session) return
    const bVal = parseFloat(inputBaseline)
    const cVal = parseFloat(inputCurrent)
    if (isNaN(bVal) || isNaN(cVal)) return

    try {
      const res = await fetch(`${API_BASE}/update-weight`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json', 
          'Authorization': `Bearer ${session.access_token}` 
        },
        body: JSON.stringify({ initial_weight: bVal, current_weight: cVal })
      })
      if (res.ok) { 
        setCurrentWeight(cVal)
        setBaselineWeight(bVal)
        setIsModalOpen(false) 
      }
    } catch (e) { console.error("Update Failure:", e) }
  }

  const trajectoryData = useMemo(() => ([
    { name: 'Base', val: parseFloat(inputBaseline) || 0 },
    { name: 'Mid', val: ((parseFloat(inputBaseline) || 0) + (parseFloat(inputCurrent) || 0)) / 2 },
    { name: 'Now', val: parseFloat(inputCurrent) || 0 }
  ]), [inputBaseline, inputCurrent])

  const weightDiff = (currentWeight - baselineWeight).toFixed(1)

  if (loading) return (
    <div className="h-screen bg-[#C1D5C0] flex flex-col items-center justify-center text-[#1E293B]">
      <Loader2 className="animate-spin mb-4" size={32} />
      <p className="font-bold tracking-widest uppercase text-xs">Registry_Sync...</p>
    </div>
  )

  return (
    <div className="min-h-screen bg-[#C1D5C0] text-[#1E293B]" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
      
      {/* 🌑 HEADER */}
      <header className="w-full bg-[#121212] border-b border-white/10 px-6 py-5 flex justify-between items-center shadow-2xl sticky top-0 z-50">
        <div className="flex items-center gap-6">
          <div className="bg-[#2D5A27] p-2 rounded-lg text-white">
            <BrainCircuit size={24} />
          </div>
          <div className="border-l border-white/20 pl-6">
            <h1 className="text-xl font-bold text-white tracking-[0.15em] uppercase">NEURO AI</h1>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Active System // 2026</p>
          </div>
        </div>

        <button onClick={() => supabase.auth.signOut()} className="px-5 py-2.5 bg-rose-950/20 border border-rose-900/40 text-rose-100 rounded-lg hover:bg-rose-900 transition-all font-bold text-[10px] uppercase tracking-widest">
          Terminate
        </button>
      </header>

      {/* 🚀 MAIN CONTENT GRID */}
      <main className="max-w-[1400px] mx-auto p-4 md:p-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-6">
          
          {/* MASS CARD */}
          <motion.div 
            onClick={() => setIsModalOpen(true)}
            whileTap={{ scale: 0.98 }} 
            className="bg-[#1E293B] p-5 rounded-xl shadow-2xl relative overflow-hidden h-48 flex flex-col justify-between text-white border border-white/10 cursor-pointer"
          >
            <div className="z-10 flex justify-between items-start">
              <div className="flex flex-col">
                <h3 className="text-sm font-bold uppercase tracking-widest text-blue-300">Body Mass</h3>
                <span className="text-[10px] opacity-40 font-sans tracking-tighter">
                  Delta: {weightDiff}kg
                </span>
              </div>
              <Weight size={18} className="opacity-40" />
            </div>
            <div className="z-10 flex items-end justify-between mb-4">
              <h2 className="text-6xl font-bold">{currentWeight}<span className="text-lg ml-1 opacity-40">KG</span></h2>
              <BarPulse color="#60A5FA" />
            </div>
            <OrganicWave color="#2563EB" />
          </motion.div>

          {/* STREAK CARD */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#E5E5D8] p-5 rounded-xl shadow-xl relative overflow-hidden h-48 flex flex-col justify-between border border-[#D4D4C5]">
            <div className="z-10 flex justify-between items-start text-[#1E293B]">
              <h3 className="text-sm font-bold uppercase tracking-widest opacity-70">Daily Streak</h3>
              <Zap size={18} className="opacity-50" />
            </div>
            <div className="z-10 flex items-end justify-between text-[#1E293B] mb-4">
              <h2 className="text-6xl font-bold">14<span className="text-lg ml-1 opacity-40">DAYS</span></h2>
              <BarPulse color="#1E293B" />
            </div>
            <OrganicWave color="white" />
          </motion.div>

          {/* EFFICIENCY CARD */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#1E293B] p-5 rounded-xl shadow-2xl relative overflow-hidden h-48 flex flex-col justify-between text-white border border-white/10">
            <div className="z-10 flex justify-between items-start">
              <h3 className="text-sm font-bold uppercase tracking-widest text-emerald-400">Efficiency</h3>
              <Activity size={18} className="opacity-40" />
            </div>
            <div className="z-10 mb-4">
              <h2 className="text-6xl font-bold">92<span className="text-lg ml-1 opacity-40">%</span></h2>
              <div className="w-full bg-white/10 h-2.5 mt-4 rounded-full overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: "92%" }} className="h-full bg-emerald-500 shadow-[0_0_15px_#10b981]" />
              </div>
            </div>
            <OrganicWave color="#10B981" />
          </motion.div>

          {/* TIME CARD */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#E5E5D8] p-5 rounded-xl shadow-xl relative overflow-hidden h-48 flex flex-col justify-between border border-[#D4D4C5]">
            <div className="z-10 flex justify-between items-start text-[#1E293B]">
              <h3 className="text-sm font-bold uppercase tracking-widest opacity-70">System Time</h3>
              <Clock size={18} className="text-rose-600 opacity-60" />
            </div>
            <div className="z-10 text-[#1E293B] mb-2">
              <h2 className="text-4xl font-bold uppercase">
                {currentTime.toLocaleDateString('en-US', { day: '2-digit', month: 'short' })}
              </h2>
              <p className="text-2xl font-bold tracking-widest mt-2">
                {currentTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}
              </p>
            </div>
            <OrganicWave color="#FB7185" />
          </motion.div>
        </div>

        {/* 🛡 FOOTER */}
        <footer className="bg-[#1E293B] p-6 rounded-2xl flex items-center justify-between text-white shadow-2xl border border-white/10">
          <div className="flex items-center gap-5">
            <ShieldCheck size={32} className="text-emerald-400"/>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.3em]">Encrypted Biometric Sync</p>
              <p className="text-[11px] opacity-40">Status: Operational // Registry: 2026</p>
            </div>
          </div>
          <button onClick={() => setIsModalOpen(true)} className="bg-white text-[#1E293B] px-12 py-4 rounded-xl font-bold uppercase text-[11px] tracking-[0.2em] hover:bg-slate-200 transition-all">
            Calibrate Metrics
          </button>
        </footer>
      </main>

      {/* 📊 THE BIG MODAL */}
      <AnimatePresence>
        {isModalOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setIsModalOpen(false)} className="absolute inset-0 bg-black/95 backdrop-blur-xl" />
            
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }} className="relative bg-[#F4F4F0] w-full max-w-5xl rounded-[2.5rem] shadow-2xl overflow-hidden">
              <div className="p-10">
                <div className="flex justify-between items-center mb-8 border-b border-slate-200 pb-6">
                  <div className="flex items-center gap-4">
                    <TrendingUp className="text-[#1E293B]" size={28} />
                    <h2 className="text-3xl font-bold uppercase tracking-tight text-[#1E293B]">Biometric Trajectory</h2>
                  </div>
                  <button onClick={() => setIsModalOpen(false)} className="p-3 bg-slate-200 rounded-full hover:bg-rose-100 transition-colors"><X size={24}/></button>
                </div>

                {/* GRAPH */}
                <div className="w-full bg-white p-8 rounded-3xl border border-slate-200 h-[380px] mb-8 shadow-inner relative overflow-hidden">
                  <div className="absolute top-6 left-10 z-10">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 mb-1">Delta Projection</p>
                    <p className="text-xl font-bold text-[#1E293B]">{inputBaseline}kg → {inputCurrent}kg</p>
                  </div>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trajectoryData}>
                      <defs>
                        <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#2563EB" stopOpacity={0.15}/><stop offset="95%" stopColor="#2563EB" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="#f0f0f0" />
                      <YAxis hide domain={['dataMin - 5', 'dataMax + 5']} />
                      <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: 'none', borderRadius: '12px', color: '#fff' }} />
                      <Area type="monotone" dataKey="val" stroke="#1E293B" strokeWidth={5} fill="url(#chartGrad)" animationDuration={1000} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* INPUTS */}
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="bg-white p-8 rounded-3xl border-2 border-slate-200 focus-within:border-blue-500 transition-all shadow-sm">
                    <label className="text-[11px] font-bold uppercase text-slate-400 block mb-3 tracking-[0.2em]">Baseline Weight (KG)</label>
                    <input type="number" step="0.1" value={inputBaseline} onChange={(e) => setInputBaseline(e.target.value)} className="w-full text-6xl font-bold outline-none bg-transparent text-[#1E293B]" />
                  </div>
                  <div className="bg-white p-8 rounded-3xl border-2 border-slate-200 focus-within:border-emerald-500 transition-all shadow-sm">
                    <label className="text-[11px] font-bold uppercase text-slate-400 block mb-3 tracking-[0.2em]">Latest Current (KG)</label>
                    <input type="number" step="0.1" value={inputCurrent} onChange={(e) => setInputCurrent(e.target.value)} className="w-full text-6xl font-bold outline-none bg-transparent text-[#1E293B]" />
                  </div>
                </div>

                <button onClick={handleUpdate} className="w-full mt-10 bg-[#121212] text-white py-6 rounded-2xl font-bold uppercase tracking-[0.5em] text-[11px] hover:bg-black transition-all shadow-2xl active:scale-[0.99] border border-white/10">
                  Commit Registry Entry
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}