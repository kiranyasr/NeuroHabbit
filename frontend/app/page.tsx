"use client"
import { useState } from 'react'

export default function AddHabitPage() {
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
    <div className="min-h-screen bg-[#F8FAFC] flex items-center justify-center p-4 font-sans selection:bg-blue-100">
      {/* Structural Blur Elements */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-400/10 rounded-full blur-[120px]"></div>
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-400/10 rounded-full blur-[120px]"></div>

      <div className="relative w-full max-w-xl bg-white/70 backdrop-blur-2xl border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] rounded-[2.5rem] p-8 md:p-12 overflow-hidden">
        
        {/* Header Section */}
        <header className="mb-12 text-center">
          <div className="inline-block px-4 py-1.5 mb-4 text-xs font-bold tracking-widest text-blue-600 uppercase bg-blue-50 rounded-full">
            Neural Engine v1.0
          </div>
          <h1 className="text-4xl font-black text-slate-900 mb-3 tracking-tight">
            Neuro<span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">Habit</span>
          </h1>
          <p className="text-slate-500 text-sm font-medium">Precision habit architecture through AI analysis.</p>
        </header>

        <div className="space-y-8">
          {/* Input Area */}
          <div className="relative">
            <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 ml-1">Input Stream</label>
            <input 
              className="w-full bg-white border border-slate-200 rounded-2xl px-6 py-5 text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/5 transition-all text-base placeholder:text-slate-300 shadow-sm" 
              onChange={(e) => setText(e.target.value)} 
              placeholder="e.g. Read for 30 mins every night"
            />
          </div>

          {/* Action Button */}
          <button 
            onClick={handleAnalyze} 
            disabled={loading || !text}
            className="w-full py-5 bg-slate-900 hover:bg-slate-800 text-white font-bold rounded-2xl shadow-xl shadow-slate-200 active:scale-[0.98] transition-all flex items-center justify-center gap-3 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </span>
            ) : "Begin Neural Analysis"}
          </button>

          {/* AI Result Card */}
          {result && (
            <div className="mt-10 p-1 bg-gradient-to-br from-slate-100 to-slate-200 rounded-[2rem]">
              <div className="bg-white p-6 rounded-[1.9rem] border border-white">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-xs font-black text-slate-900 uppercase tracking-[0.2em]">Verified Data</h3>
                  <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-100">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
                    SYNCED
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-slate-50/50 rounded-2xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Activity</p>
                    <p className="font-bold text-slate-800 capitalize">{result.saved_data?.activity || "—"}</p>
                  </div>
                  <div className="p-4 bg-slate-50/50 rounded-2xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Duration</p>
                    <p className="font-bold text-slate-800">{result.saved_data?.duration || 0}m</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}