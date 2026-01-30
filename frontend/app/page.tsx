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
  const [risks, setRisks] = useState<any[]>([])
  const [result, setResult] = useState<any>(null)
  const [nudge, setNudge] = useState<{style: string, message: string} | null>(null)
  const [history, setHistory] = useState<any[]>([])
  const [analyticsData, setAnalyticsData] = useState<any[]>([])

  const fetchData = async () => {
    try {
      const res = await fetch("http://localhost:8000/get-habits")
      if (!res.ok) throw new Error("Backend Offline");
      const data = await res.json()
      const habits = data.habits || []
      setHistory(habits)
      
      const formatted = habits.slice(0, 5).map((h: any) => ({
        name: h.activity || "UNKNOWN",
        risk: h.risk_score || 0 
      }))
      setAnalyticsData(formatted)
    } catch (error) {
      console.error("System Fetch Error:", error)
    }
  }

  useEffect(() => {
    fetchData();
    fetch("http://localhost:8000/predict-risk")
      .then(res => res.ok ? res.json() : null)
      .then(data => { if(data?.risks) setRisks(data.risks) })
      .catch(() => console.log("Predictive engine standby..."));
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
      if (!res.ok) throw new Error("Failed to deploy habit");
      const data = await res.json()
      
      setResult(data)
      setNudge(data.nudge)
      fetchData() 
      setText("")
    } catch (err) {
      console.error("Submission Error:", err);
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#1A2238] selection:bg-[#E30613]/20" 
         style={{ fontFamily: '"Times New Roman", Times, serif' }}>
      
      {/* 🏛️ TOP NAVIGATION */}
      <nav className="bg-[#1A2238] text-white px-8 py-4 flex justify-between items-center sticky top-0 z-50 shadow-md">
        <div className="flex items-center gap-3">
          <div className="bg-[#E30613] w-8 h-8 flex items-center justify-center rounded-sm">
             <span className="text-xs font-bold">NH</span>
          </div>
          <h1 className="text-lg font-bold tracking-widest uppercase">
            NEURO<span className="text-[#E30613]">HABIT</span>
          </h1>
        </div>
        <div className="flex gap-6 text-[10px] font-bold uppercase tracking-widest text-slate-400">
          <span className="text-white border-b-2 border-[#E30613]">Live Tracker</span>
          <span>Behavioral Logs</span>
          <span>System Stats</span>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-10 space-y-6">
        
        {/* 🧠 DAY 12: AI NUDGE HUD (FIXED: NO SLANTING) */}
        <AnimatePresence>
          {nudge && (
            <motion.div 
              initial={{ opacity: 0, x: -20 }} 
              animate={{ opacity: 1, x: 0 }}
              className="bg-[#1A2238] border-l-4 border-[#E30613] p-6 rounded-r-xl shadow-lg relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-2 opacity-10">
                <span className="text-4xl text-white font-bold">{nudge.style[0]}</span>
              </div>
              <h5 className="text-[10px] font-bold text-[#E30613] uppercase tracking-[0.3em] mb-2">
                AI PROTOCOL: {nudge.style} NUDGE
              </h5>
              {/* Removed 'italic' class here */}
              <p className="text-white text-sm font-medium leading-relaxed not-italic">
                "{nudge.message}"
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 🧠 SYSTEM STATUS HUD */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {risks.map((r, i) => r.risk > 40 && (
            <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="bg-white border-l-4 border-[#E30613] p-4 shadow-sm flex justify-between items-center rounded-r-lg border border-slate-100">
              <p className="text-xs font-bold uppercase tracking-tight">Predicted Friction: {r.habit}</p>
              <span className="bg-[#1A2238] text-white px-2 py-1 text-[10px] font-bold">{r.risk}% RISK</span>
            </motion.div>
          ))}
        </div>

        {/* 📝 HABIT ENTRY CONSOLE */}
        <div className="bg-[#1A2238] p-8 rounded-tr-[2rem] rounded-bl-[2rem] shadow-xl relative overflow-hidden">
           <div className="absolute top-0 right-0 w-24 h-24 bg-[#E30613] rotate-45 translate-x-12 -translate-y-12 opacity-10"></div>
           <div className="relative z-10 space-y-4">
              <div className="flex items-center gap-3">
                <div className="h-0.5 w-8 bg-[#E30613]"></div>
                <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.3em]">Initialize Habit Sequence</h3>
              </div>
              <div className="flex flex-col md:flex-row gap-3">
                <input 
                  className="flex-1 bg-white border border-slate-200 rounded-lg px-5 py-3 text-base font-bold text-[#1A2238] outline-none uppercase" 
                  style={{ fontFamily: '"Times New Roman", Times, serif' }}
                  placeholder="E.G. GYM FOR 45 MINUTES"
                  value={text}
                  onChange={(e) => setText(e.target.value.toUpperCase())}
                />
                <button onClick={handleAnalyze} className="bg-[#E30613] text-white px-8 py-3 rounded-lg text-xs font-bold uppercase tracking-widest hover:bg-[#C10510] transition-all">
                  {loading ? "PARSING..." : "DEPLOY HABIT"}
                </button>
              </div>
           </div>
        </div>

        <RiskChart data={analyticsData} />

        {/* 📋 NEURAL HABIT MANIFEST */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
           <div className="px-6 py-3 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
              <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">Neural Habit Manifest</h4>
              <span className="text-[9px] text-[#E30613] font-bold uppercase tracking-widest">Live Sync</span>
           </div>
           <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50/50">
                  <th className="px-6 py-3 text-[9px] font-bold text-slate-400 uppercase tracking-widest">Protocol ID</th>
                  <th className="px-6 py-3 text-[9px] font-bold text-slate-400 uppercase tracking-widest">Activity</th>
                  <th className="px-6 py-3 text-[9px] font-bold text-slate-400 uppercase tracking-widest text-right">Risk Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {history.map((h, i) => (
                  <tr key={i} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">#NH-{h.id.toString().slice(0,4)}</td>
                    <td className="px-6 py-4 text-xs font-bold text-[#1A2238] uppercase tracking-tight">{h.activity}</td>
                    <td className="px-6 py-4 text-right">
                       <span className={`text-[10px] font-bold px-2 py-1 rounded-sm ${
                         h.risk_score > 60 ? 'bg-red-100 text-[#E30613]' : 
                         h.risk_score > 30 ? 'bg-amber-100 text-amber-600' : 
                         'bg-emerald-100 text-emerald-600'
                       }`}>
                        {h.risk_score}% RISK
                       </span>
                    </td>
                  </tr>
                ))}
              </tbody>
           </table>
        </div>
      </main>
    </div>
  )
}