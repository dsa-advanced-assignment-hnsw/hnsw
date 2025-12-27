import React, { useState } from 'react';
import { Search, PlusCircle, Database, ChevronRight, ChevronDown, Trash2, Minus, Plus } from 'lucide-react';

interface ControlsProps {
    onInit: (max: number, m: number, ef: number, dim: number, initCount: number) => void;
    onInsert: (customVector?: number[]) => void;
    onSearch: (customVector: number[] | undefined, k: number, ef: number) => void;
    onClear: () => void;
    speed: number;
    setSpeed: (s: number) => void;
    loading: boolean;
}

const NumberInput = ({ label, value, onChange, min = 1, max, step = 1 }: {
    label: string, value: number, onChange: (val: number) => void, min?: number, max?: number, step?: number
}) => {
    const handleDecrement = () => {
        if (value > min) onChange(value - step);
    };
    const handleIncrement = () => {
        if (!max || value < max) onChange(value + step);
    };

    return (
        <div className="flex flex-col gap-1">
            <label className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 tracking-wider">{label}</label>
            <div className="flex items-center h-8 bg-white/50 dark:bg-slate-900/50 rounded-md border border-slate-300 dark:border-slate-700 overflow-hidden transition-colors focus-within:border-cyan-500">
                <button
                    onClick={handleDecrement}
                    className="w-8 h-full flex items-center justify-center text-slate-500 hover:text-rose-500 hover:bg-black/5 transition-colors"
                >
                    <Minus size={12} />
                </button>
                <input
                    type="number"
                    value={value}
                    onChange={(e) => onChange(Number(e.target.value))}
                    className="flex-1 w-full bg-transparent text-center text-xs font-mono font-bold text-slate-800 dark:text-cyan-400 focus:outline-none appearance-none [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    min={min}
                    max={max}
                    step={step}
                />
                <button
                    onClick={handleIncrement}
                    className="w-8 h-full flex items-center justify-center text-slate-500 hover:text-emerald-500 hover:bg-black/5 transition-colors"
                >
                    <Plus size={12} />
                </button>
            </div>
        </div>
    );
};

export const Controls: React.FC<ControlsProps> = ({ onInit, onInsert, onSearch, onClear, speed, setSpeed, loading }) => {
    const [initParams, setInitParams] = useState({ max: 50, m: 3, ef: 10, dim: 3, count: 20 });
    const [searchParams, setSearchParams] = useState({ k: 1, ef: 10 });
    const [showInit, setShowInit] = useState(true);
    const [showActions, setShowActions] = useState(true);
    const [vectorInput, setVectorInput] = useState<string>("");

    const parseVector = (str: string): number[] | undefined => {
        if (!str.trim()) return undefined;
        try {
            const arr = str.split(',').map(n => parseFloat(n.trim()));
            if (arr.some(isNaN)) return undefined;
            return arr;
        } catch {
            return undefined;
        }
    };

    const SectionHeader = ({ title, isOpen, toggle }: { title: string, isOpen: boolean, toggle: () => void }) => (
        <button
            onClick={toggle}
            className="flex items-center justify-between w-full py-2 px-1 text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-cyan-500 transition-colors"
        >
            {title}
            {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
    );

    return (
        <div className="flex flex-col gap-4 text-sm w-full">
            <div className="bg-white/40 dark:bg-slate-800/40 rounded-lg p-3 border border-slate-200 dark:border-white/5 shadow-sm">
                <SectionHeader title="Graph Settings" isOpen={showInit} toggle={() => setShowInit(!showInit)} />
                {showInit && (
                    <div className="mt-3 grid grid-cols-2 gap-3">
                        <NumberInput label="Dim" value={initParams.dim} onChange={(v) => setInitParams({ ...initParams, dim: v })} min={1} max={10} />
                        <NumberInput label="Max Elements" value={initParams.max} onChange={(v) => setInitParams({ ...initParams, max: v })} />
                        <NumberInput label="M (Neighbors)" value={initParams.m} onChange={(v) => setInitParams({ ...initParams, m: v })} min={2} />
                        <NumberInput label="ef (Construction)" value={initParams.ef} onChange={(v) => setInitParams({ ...initParams, ef: v })} min={1} />
                        <NumberInput label="Count" value={initParams.count} onChange={(v) => setInitParams({ ...initParams, count: v })} min={1} />
                        <button
                            onClick={() => onInit(initParams.max, initParams.m, initParams.ef, initParams.dim, initParams.count)}
                            disabled={loading}
                            className="col-span-2 flex items-center justify-center gap-2 mt-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:opacity-90 disabled:opacity-50 text-white py-2 rounded-md font-bold transition-all active:scale-95"
                        >
                            <Database size={16} />
                            {loading ? 'Initializing...' : 'Reset & Init'}
                        </button>
                    </div>
                )}
            </div>

            <div className="bg-white/40 dark:bg-slate-800/40 rounded-lg p-3 border border-slate-200 dark:border-white/5 shadow-sm">
                <SectionHeader title="Operations" isOpen={showActions} toggle={() => setShowActions(!showActions)} />
                {showActions && (
                    <div className="flex flex-col gap-4 mt-3">
                        <div className="flex flex-col gap-1">
                            <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider">
                                <span>Animation Speed</span>
                                <span className="text-indigo-600 dark:text-cyan-400">{speed}ms</span>
                            </div>
                            <input
                                type="range" min="50" max="2000" step="50"
                                value={speed} onChange={e => setSpeed(+e.target.value)}
                                className="w-full h-1 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600 transition-colors"
                            />
                        </div>
                        <div className="flex flex-col gap-1">
                            <label className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 tracking-wider">Custom Vector</label>
                            <input
                                type="text"
                                placeholder="e.g. 0.5, 1.2, -0.3"
                                value={vectorInput}
                                onChange={e => setVectorInput(e.target.value)}
                                className="w-full h-8 bg-white/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-700 rounded px-2 py-1 text-slate-800 dark:text-white text-xs placeholder-slate-400 focus:border-indigo-500 focus:outline-none transition-colors"
                            />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => onInsert(parseVector(vectorInput))}
                                disabled={loading}
                                className="flex items-center justify-center gap-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600 py-2 rounded-md font-semibold transition-all active:scale-95"
                            >
                                <PlusCircle size={16} /> Insert
                            </button>
                            <button
                                onClick={() => onSearch(parseVector(vectorInput), searchParams.k, searchParams.ef)}
                                disabled={loading}
                                className="flex items-center justify-center gap-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600 py-2 rounded-md font-semibold transition-all active:scale-95"
                            >
                                <Search size={16} /> Search
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <NumberInput label="K" value={searchParams.k} onChange={(v) => setSearchParams({ ...searchParams, k: v })} min={1} />
                            <NumberInput label="EF (Search)" value={searchParams.ef} onChange={(v) => setSearchParams({ ...searchParams, ef: v })} min={1} />
                        </div>
                        <button
                            onClick={onClear}
                            className="w-full flex items-center justify-center gap-2 text-xs font-bold text-slate-400 hover:text-rose-500 py-2 border-t border-slate-200 dark:border-white/5 transition-colors mt-2"
                        >
                            <Trash2 size={14} /> Clear Visualization
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};
