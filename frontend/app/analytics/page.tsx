"use client"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// This is a "Summary" of your 5,000 rows for the graph
const weeklyData = [
  { day: 'Mon', success: 82 },
  { day: 'Tue', success: 78 },
  { day: 'Wed', success: 65 }, // Mid-week slump!
  { day: 'Thu', success: 68 },
  { day: 'Fri', success: 85 },
  { day: 'Sat', success: 45 }, // Weekend failure pattern!
  { day: 'Sun', success: 42 },
];

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-[#F8FAFC] p-8 font-sans">
      <div className="max-w-6xl mx-auto">
        <header className="mb-10">
          <h1 className="text-3xl font-black text-slate-900 tracking-tight">Neural <span className="text-blue-600">Insights</span></h1>
          <p className="text-slate-500 font-medium">Visualizing behavioral patterns from 5,000 data points.</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Avg. Success</p>
            <p className="text-2xl font-black text-slate-900">66.5%</p>
          </div>
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Weakest Link</p>
            <p className="text-2xl font-black text-red-500">Sundays</p>
          </div>
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Strongest Habit</p>
            <p className="text-2xl font-black text-emerald-500">Deep Work</p>
          </div>
        </div>

        <div className="bg-white p-8 rounded-[2.5rem] border border-slate-200 shadow-sm">
          <h3 className="text-xs font-bold text-slate-400 uppercase mb-8 tracking-widest">Consistency Curve (Weekly Avg)</h3>
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fill: '#94A3B8', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#94A3B8', fontSize: 12}} />
                <Tooltip contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}} />
                <Line type="monotone" dataKey="success" stroke="#2563EB" strokeWidth={4} dot={{ r: 6, fill: '#2563EB', strokeWidth: 2, stroke: '#fff' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}