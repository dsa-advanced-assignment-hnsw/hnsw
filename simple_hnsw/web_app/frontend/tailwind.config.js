/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                slate: {
                    850: '#151e2e',
                    900: '#0f172a',
                    950: '#020617',
                },
                neon: {
                    cyan: '#00f3ff',
                    pink: '#ff00ff',
                    purple: '#bc13fe',
                    orange: '#ff4d00',
                },
                glass: {
                    low: 'rgba(255, 255, 255, 0.05)',
                    medium: 'rgba(255, 255, 255, 0.1)',
                    high: 'rgba(255, 255, 255, 0.2)',
                    dark: 'rgba(0, 0, 0, 0.4)',
                }
            },
            boxShadow: {
                'neon-cyan': '0 0 10px #00f3ff, 0 0 20px rgba(0, 243, 255, 0.4)',
                'neon-pink': '0 0 10px #ff00ff, 0 0 20px rgba(255, 0, 255, 0.4)',
                'glass-inset': 'inset 0 0 20px rgba(255, 255, 255, 0.05)',
            },
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui'],
                mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
            }
        },
    },
    plugins: [],
}
