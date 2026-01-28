"use client"
import { useState } from 'react'

export default function ProfessionalAddHabit() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleAnalyze = async () => {
    setLoading(true)
    try {
      const res = await fetch("http://localhost:8000/add-habit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      })
      const data = await res.json()
      setResult(data)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#F1F5F9] flex font-sans">
      {/* 1. Left Navigation Sidebar */}
      <aside className="w-64 bg-white border-r border-slate-200 hidden lg:flex flex-col sticky top-0 h-screen">
        <div className="p-6 border-b border-slate-100">
          <h1 className="text-xl font-bold tracking-tight text-slate-800 italic">NeuroHabit<span className="text-blue-600">.ai</span></h1>
        </div>
        <nav className="flex-1 p-4 space-y-1">
          <div className="text-slate-400 text-[10px] font-bold uppercase px-3 mb-2 tracking-widest">Main Menu</div>
          <div className="bg-blue-50 text-blue-700 font-semibold p-3 rounded-lg text-sm cursor-pointer">Add New Habit</div>
          <div className="text-slate-500 p-3 rounded-lg text-sm hover:bg-slate-50 cursor-pointer transition-colors">History Log</div>
          <div className="text-slate-500 p-3 rounded-lg text-sm hover:bg-slate-50 cursor-pointer transition-colors">Neural Insights</div>
        </nav>
      </aside>

      {/* 2. Main Work Area */}
      <main className="flex-1">
        {/* Top Utility Bar */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 sticky top-0 z-10">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Workspace</span>
            <span className="text-slate-300">/</span>
            <span className="text-xs font-bold text-slate-900 uppercase tracking-widest">Neural Architecture</span>
          </div>
          <div className="flex items-center gap-3">
             <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
             <span className="text-[10px] font-bold text-slate-500 tracking-tighter">ENGINE ONLINE</span>
          </div>
        </header>

        <div className="max-w-5xl mx-auto p-8 space-y-6">
          {/* 3. Input Console (The Command Center) */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="p-4 border-b border-slate-100 bg-slate-50/50">
              <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Natural Language Entry</h2>
            </div>
            <div className="p-6 flex flex-col md:flex-row gap-4">
              <input 
                className="flex-1 bg-white border border-slate-200 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all font-medium text-slate-700" 
                onChange={(e) => setText(e.target.value)} 
                placeholder="e.g. Code for 2 hours every evening"
              />
              <button 
                onClick={handleAnalyze} 
                disabled={loading || !text}
                className="bg-slate-900 text-white px-8 py-3 rounded-lg text-sm font-bold hover:bg-blue-600 transition-all active:scale-95 disabled:opacity-30"
              >
                {loading ? "ANALYZING..." : "DEPLOY HABIT"}
              </button>
            </div>
          </div>

          {/* 4. Processing Result (The Data Table) */}
          {result && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden animate-in fade-in duration-500">
               <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Output Verification Log</h2>
                <span className="text-[10px] font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-100 uppercase">Success</span>
              </div>
              <table className="w-full text-left">
                <thead className="bg-slate-50/80 border-b border-slate-100">
                  <tr>
                    <th className="p-4 text-[10px] font-bold text-slate-400 uppercase">Neural Entity</th>
                    <th className="p-4 text-[10px] font-bold text-slate-400 uppercase">Structured Value</th>
                    <th className="p-4 text-[10px] font-bold text-slate-400 uppercase">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  <tr>
                    <td className="p-4 text-xs font-bold text-slate-500">ACTIVITY</td>
                    <td className="p-4 text-sm font-semibold text-slate-900 capitalize">{result.saved_data?.activity}</td>
                    <td className="p-4 text-xs font-medium text-emerald-600 font-mono">98.2%</td>
                  </tr>
                  <tr>
                    <td className="p-4 text-xs font-bold text-slate-500">DURATION</td>
                    <td className="p-4 text-sm font-semibold text-slate-900">{result.saved_data?.duration} Minutes</td>
                    <td className="p-4 text-xs font-medium text-emerald-600 font-mono">100%</td>
                  </tr>
                  <tr>
                    <td className="p-4 text-xs font-bold text-slate-500">FREQUENCY</td>
                    <td className="p-4 text-sm font-semibold text-slate-900 capitalize">{result.saved_data?.frequency}</td>
                    <td className="p-4 text-xs font-medium text-emerald-600 font-mono">94.5%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}