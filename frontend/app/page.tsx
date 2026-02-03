"use client"
import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { createClient } from '@supabase/supabase-js'
import { 
  Weight, Zap, Activity, Clock, LogOut, ShieldCheck, 
  BrainCircuit
} from 'lucide-react'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
const supabase = createClient(supabaseUrl, supabaseKey)

/* ================= 📊 HIGH-VISIBILITY ANIMATIONS ================= */

const OrganicWave = ({ color, opacity = "0.4" }: { color: string, opacity?: string }) => (
  <div className={`absolute bottom-0 left-0 w-full h-24 pointer-events-none`} style={{ opacity }}>
    <svg className="w-[200%] h-full animate-wave-fast" viewBox="0 0 1000 100" preserveAspectRatio="none">
      <path d="M0,60 C150,110 350,10 500,60 C650,110 850,10 1000,60 L1000,100 L0,100 Z" fill={color} />
    </svg>
    <style jsx>{`
      @keyframes wave-fast { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
      .animate-wave-fast { animation: wave-fast 3s linear infinite; }
    `}</style>
  </div>
)

const BarPulse = ({ color }: { color: string }) => (
  <div className="flex gap-1.5 h-16 items-end opacity-80">
    {[0.4, 0.9, 0.5, 0.8, 0.6].map((h, i) => (
      <motion.div
        key={i}
        animate={{ height: [`${h * 100}%`, `${(h > 0.5 ? h-0.4 : h+0.4) * 100}%`, `${h * 100}%`] }}
        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.1 }}
        className="w-3 rounded-t-md"
        style={{ backgroundColor: color }}
      />
    ))}
  </div>
)

/* ================= 🚀 NEURO AI MAIN UI ================= */

export default function NeuroAI_Dashboard() {
  const [session, setSession] = useState<any>(null)
  const [currentTime, setCurrentTime] = useState(new Date())
  const stats = { weight: "76.2", streak: "14", efficiency: "92" }

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    supabase.auth.getSession().then(({ data }) => setSession(data.session))
    return () => clearInterval(timer)
  }, [])

  if (!session) return <div className="h-screen bg-[#C1D5C0] flex items-center justify-center font-serif font-bold text-[#1E293B]">SYSTEM_RECALL...</div>

  return (
    <div className="min-h-screen bg-[#C1D5C0] text-[#1E293B] font-serif" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
      
      {/* 🌑 COMMAND HEADER */}
      <header className="w-full bg-[#121212] border-b border-white/5 px-6 py-4 flex justify-between items-center shadow-2xl sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="bg-[#2D5A27] p-2 rounded-lg text-white shadow-[0_0_15px_rgba(45,90,39,0.4)]">
            <BrainCircuit size={22} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight uppercase">NEURO AI</h1>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Protocol Active</p>
            </div>
          </div>
        </div>

        <button 
          onClick={() => supabase.auth.signOut()} 
          className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 text-white rounded-lg hover:bg-rose-900 transition-all font-bold text-[10px] uppercase tracking-widest"
        >
          <LogOut size={14} />
          <span>Logout</span>
        </button>
      </header>

      {/* 🚀 MAIN CONTENT GRID */}
      <main className="max-w-[1400px] mx-auto p-4 md:p-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
          
          {/* MASS CARD */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#1E293B] p-5 rounded-xl shadow-2xl relative overflow-hidden h-52 flex flex-col justify-between text-white border border-white/10">
            <div className="z-10 flex justify-between items-start">
              <h3 className="text-sm font-bold uppercase tracking-widest text-blue-300">Body Mass</h3>
              <Weight size={18} className="opacity-40" />
            </div>
            <div className="z-10 flex items-end justify-between mb-2">
              <h2 className="text-6xl font-bold">{stats.weight}<span className="text-lg ml-1 opacity-40">KG</span></h2>
              <BarPulse color="#60A5FA" />
            </div>
            <OrganicWave color="#2563EB" />
          </motion.div>

          {/* STREAK CARD - ANIMATION FIXED HERE */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#E5E5D8] p-5 rounded-xl shadow-xl relative overflow-hidden h-52 flex flex-col justify-between border border-[#D4D4C5]">
            <div className="z-10 flex justify-between items-start text-[#1E293B]">
              <h3 className="text-sm font-bold uppercase tracking-widest opacity-70">Daily Streak</h3>
              <Zap size={18} className="opacity-50" />
            </div>
            <div className="z-10 flex items-end justify-between text-[#1E293B] mb-2">
              <h2 className="text-6xl font-bold">{stats.streak}<span className="text-lg ml-1 opacity-40">DAYS</span></h2>
              <BarPulse color="#1E293B" />
            </div>
            {/* Wave changed to Navy for visibility on Beige */}
            <OrganicWave color="#1E293B" opacity="0.15" />
          </motion.div>

          {/* EFFICIENCY CARD */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#1E293B] p-5 rounded-xl shadow-2xl relative overflow-hidden h-52 flex flex-col justify-between text-white border border-white/10">
            <div className="z-10 flex justify-between items-start">
              <h3 className="text-sm font-bold uppercase tracking-widest text-emerald-400">Efficiency</h3>
              <Activity size={18} className="opacity-40" />
            </div>
            <div className="z-10 mb-2">
              <h2 className="text-6xl font-bold">{stats.efficiency}<span className="text-lg ml-1 opacity-40">%</span></h2>
              <div className="w-full bg-white/10 h-3 mt-4 rounded-full overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: "92%" }} className="h-full bg-emerald-500 shadow-[0_0_15px_#10b981]" />
              </div>
            </div>
            <OrganicWave color="#10B981" />
          </motion.div>

          {/* TIME CARD */}
          <motion.div whileTap={{ scale: 0.98 }} className="bg-[#E5E5D8] p-5 rounded-xl shadow-xl relative overflow-hidden h-52 flex flex-col justify-between border border-[#D4D4C5]">
            <div className="z-10 flex justify-between items-start text-[#1E293B]">
              <h3 className="text-sm font-bold uppercase tracking-widest opacity-70">System Time</h3>
              <Clock size={18} className="text-rose-600 opacity-60" />
            </div>
            <div className="z-10 text-[#1E293B] mb-2">
              <h2 className="text-4xl font-bold uppercase leading-none">
                {currentTime.toLocaleDateString('en-US', { day: '2-digit', month: 'short' })}
              </h2>
              <p className="text-2xl font-bold tracking-widest mt-2">
                {currentTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </p>
            </div>
            {/* Wave changed to Rose for visibility on Beige */}
            <OrganicWave color="#FB7185" opacity="0.2" />
          </motion.div>

        </div>
      </main>
    </div>
  )
}