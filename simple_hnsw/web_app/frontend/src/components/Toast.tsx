import React, { useEffect, useState } from 'react';
import { X, CheckCircle, AlertCircle, Info } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info';

export interface ToastMessage {
    id: string;
    type: ToastType;
    message: string;
}

interface ToastProps {
    toasts: ToastMessage[];
    removeToast: (id: string) => void;
}

export const ToastContainer: React.FC<ToastProps> = ({ toasts, removeToast }) => {
    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
            {toasts.map((toast) => (
                <ToastItem key={toast.id} toast={toast} removeToast={removeToast} />
            ))}
        </div>
    );
};

const ToastItem: React.FC<{ toast: ToastMessage; removeToast: (id: string) => void }> = ({ toast, removeToast }) => {
    const [isExiting, setIsExiting] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => {
            handleClose();
        }, 4000);
        return () => clearTimeout(timer);
    }, []);

    const handleClose = () => {
        setIsExiting(true);
        setTimeout(() => removeToast(toast.id), 300); // Wait for exit animation
    };

    const icons = {
        success: <CheckCircle size={20} className="text-neon-cyan" />,
        error: <AlertCircle size={20} className="text-neon-pink" />,
        info: <Info size={20} className="text-neon-purple" />
    };

    const borders = {
        success: 'border-neon-cyan/30',
        error: 'border-neon-pink/30',
        info: 'border-neon-purple/30'
    };

    return (
        <div
            className={`
        pointer-events-auto
        flex items-center gap-3 px-4 py-3 rounded-lg border backdrop-blur-md shadow-lg
        bg-glass-dark text-white min-w-[300px]
        transition-all duration-300 transform
        ${borders[toast.type]}
        ${isExiting ? 'opacity-0 translate-x-full' : 'opacity-100 translate-x-0'}
      `}
        >
            {icons[toast.type]}
            <p className="flex-1 text-sm font-medium font-sans">{toast.message}</p>
            <button onClick={handleClose} className="text-slate-400 hover:text-white transition-colors">
                <X size={16} />
            </button>
        </div>
    );
};
