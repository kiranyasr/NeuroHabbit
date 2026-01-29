"use client"
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function UnifiedStableConsole() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [risks, setRisks] = useState<any[]>([])
  const [result, setResult] = useState<any>(null)

  useEffect(() => {
    fetch("http://localhost:8000/predict-risk")
      .then(res => res.json())
      .then(data => { if(data.risks) setRisks(data.risks) });
  }, []);

  const handleAnalyze = async () => {
    setLoading(true)
    try {
      const res = await fetch("http://localhost:8000/add-habit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.toUpperCase() })
      })
      const data = await res.json()
      setResult(data)
    } finally {
      setLoading(false)
    }
  }

  return (
    // 🖋️ Applying Times New Roman globally and removing all slanting (italics)
    <div className="min-h-screen bg-[#F8FAFC] text-[#1A2238] selection:bg-[#E30613]/20" 
         style={{ fontFamily: '"Times New Roman", Times, serif' }}>
      
      {/* 🏛️ TOP NAVIGATION (No Slanting) */}
      <nav className="bg-[#1A2238] text-white px-8 py-4 flex justify-between items-center sticky top-0 z-50 shadow-md">
        <div className="flex items-center gap-3">
          <div className="bg-[#E30613] w-8 h-8 flex items-center justify-center rounded-sm">
             <span className="text-xs font-bold">NH</span>
          </div>
          {/* 🚀 Removed italic from logo */}
          <h1 className="text-lg font-bold tracking-widest uppercase">
            NEURO<span className="text-[#E30613]">HABIT</span>
          </h1>
        </div>
        <div className="flex gap-6 text-[10px] font-bold uppercase tracking-widest text-slate-400">
          <span className="text-white border-b-2 border-[#E30613] cursor-pointer">Live Tracker</span>
          <span className="hover:text-[#E30613] transition-all cursor-pointer">Behavioral Logs</span>
          <span className="hover:text-[#E30613] transition-all cursor-pointer">System Stats</span>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-10 space-y-6">
        
        {/* 🧠 COMPACT SYSTEM HUD (No Slanting) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {risks.map((r, i) => r.risk > 40 && (
            <motion.div 
              key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="bg-white border-l-4 border-[#E30613] p-4 shadow-sm flex justify-between items-center rounded-r-lg border border-slate-100">
              <p className="text-xs font-bold uppercase tracking-tight">Predicted Friction: {r.habit}</p>
              {/* 🚀 Removed italic from risk status */}
              <span className="bg-[#1A2238] text-white px-2 py-1 text-[10px] font-bold">{r.risk}% RISK</span>
            </motion.div>
          ))}
        </div>

        {/* 📝 HABIT ENTRY CONSOLE (No Slanting) */}
        <div className="bg-[#1A2238] p-8 rounded-tr-[2rem] rounded-bl-[2rem] shadow-xl relative overflow-hidden">
           <div className="absolute top-0 right-0 w-24 h-24 bg-[#E30613] rotate-45 translate-x-12 -translate-y-12 opacity-10"></div>
           
           <div className="relative z-10 space-y-4">
              <div className="flex items-center gap-3">
                <div className="h-0.5 w-8 bg-[#E30613]"></div>
                <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.3em]">Initialize Habit Sequence</h3>
              </div>
              
              <div className="flex flex-col md:flex-row gap-3">
                <input 
                  className="flex-1 bg-white border border-slate-200 rounded-lg px-5 py-3 text-base font-bold text-[#1A2238] outline-none placeholder:text-slate-300 uppercase shadow-inner" 
                  style={{ fontFamily: '"Times New Roman", Times, serif' }}
                  placeholder="E.G. GYM FOR 45 MINUTES"
                  value={text.toUpperCase()}
                  onChange={(e) => setText(e.target.value)}
                />
                <button 
                  onClick={handleAnalyze}
                  className="bg-[#E30613] text-white px-8 py-3 rounded-lg text-xs font-bold uppercase tracking-widest shadow-lg shadow-red-600/30 hover:bg-[#C10510] active:scale-95 transition-all"
                >
                  {loading ? "PARSING..." : "DEPLOY HABIT"}
                </button>
              </div>
           </div>
        </div>

        {/* 📊 REFINED EXTRACTION HUD (No Slanting) */}
        <AnimatePresence>
          {result && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
              className="bg-white border border-slate-200 rounded-[2rem] shadow-xl overflow-hidden max-w-2xl mx-auto">
               <div className="bg-[#1A2238] py-2.5 text-center border-b border-[#E30613]">
                  <span className="text-[10px] font-bold text-white uppercase tracking-[0.4em]">Behavioral Extraction Successful</span>
               </div>
               <div className="grid grid-cols-2 divide-x divide-slate-100">
                  <div className="p-6 text-center">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Identifier</p>
                    {/* 🚀 Removed italic from activity text */}
                    <p className="text-2xl font-bold text-[#1A2238] uppercase tracking-tighter leading-none" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
                      {result.saved_data.activity}
                    </p>
                  </div>
                  <div className="p-6 text-center bg-slate-50/50">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Duration</p>
                    <div className="flex justify-center items-baseline gap-2">
                      <span className="text-4xl font-bold text-[#E30613] tracking-tighter leading-none" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
                        {result.saved_data.duration}
                      </span>
                      <span className="bg-[#1A2238] text-white text-[9px] font-bold px-2 py-0.5 rounded uppercase tracking-tighter shadow-sm">Minutes</span>
                    </div>
                  </div>
               </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 🏢 PERFORMANCE METRICS (No Slanting) */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-6 border-t border-slate-200">
           <div className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm text-center">
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Neural Accuracy</h4>
             <p className="text-xl font-bold text-[#1A2238]">99.8% <span className="text-[10px] text-[#E30613]">SYNC</span></p>
           </div>
           <div className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm text-center">
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">System Load</h4>
             <p className="text-xl font-bold text-[#1A2238]">0.2ms <span className="text-[10px] text-[#E30613]">LATENCY</span></p>
           </div>
           <div className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm text-center">
             <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Active Monitors</h4>
             <p className="text-xl font-bold text-[#1A2238]">Active <span className="text-[10px] text-[#E30613]">LIVE</span></p>
           </div>
        </section>

      </main>
    </div>
  )
}