// Theme configuration for light and dark modes
export type ThemeMode = 'light' | 'dark';

export interface Theme {
    mode: ThemeMode;
    background: string;
    canvas: string;
    text: {
        primary: string;
        secondary: string;
        outline: string;
    };
    nodes: {
        getLayerColor: (level: number, maxLevel: number) => string;
        activeColor: string;
        candidateColor: string;
    };
    edges: {
        permanent: string;
        permanent_opacity: number;
        vertical: string;
        vertical_opacity: number;
        persistent: string;
        selected: string;
        considered: string;
        rejected: string;
    };
    planes: {
        getColor: (level: number, maxLevel: number) => string;
        opacity: number;
        gridPrimary: string;
        gridSecondary: string;
    };
    lighting: {
        ambient: number;
        directional1: number;
        directional2: number;
        hemisphereTop: string;
        hemisphereBottom: string;
    };
}

// Light theme - matching Python visualization
export const lightTheme: Theme = {
    mode: 'light',
    background: 'rgb(249, 250, 251)',
    canvas: '#ffffff',
    text: {
        primary: '#000000',
        secondary: '#666666',
        outline: '#ffffff'
    },
    nodes: {
        getLayerColor: (level: number, maxLevel: number) => {
            // Match Python matplotlib.cm.Blues
            // Low level -> lighter blue, High level -> darker blue
            const t = 0.2 + 0.8 * (level / (maxLevel || 1)); // 0.2 to 1.0 range
            // Intepolate between Light Blue (#DEEBF7) and Dark Blue (#08306B)
            const r = Math.floor(222 * (1 - t) + 8 * t);
            const g = Math.floor(235 * (1 - t) + 48 * t);
            const b = Math.floor(247 * (1 - t) + 107 * t);
            return `rgb(${r}, ${g}, ${b})`;
        },
        activeColor: '#22c55e',
        candidateColor: '#22c55e'
    },
    edges: {
        permanent: '#999999',
        permanent_opacity: 0.25,
        vertical: '#cccccc',
        vertical_opacity: 0.4,
        persistent: '#3b82f6',
        selected: '#22c55e',
        considered: '#f97316',
        rejected: '#ef4444'
    },
    planes: {
        getColor: (level: number, maxLevel: number) => {
            const intensity = 0.4 + 0.5 * (level / (maxLevel || 1));
            const blue = Math.floor(intensity * 255);
            return `rgb(${Math.floor(blue * 0.3)}, ${Math.floor(blue * 0.5)}, ${blue})`;
        },
        opacity: 0.08,
        gridPrimary: '#cccccc',
        gridSecondary: '#e5e5e5'
    },
    lighting: {
        ambient: 0.7,
        directional1: 0.8,
        directional2: 0.4,
        hemisphereTop: '#ffffff',
        hemisphereBottom: '#999999'
    }
};

// Dark theme - inverted with warm accents
export const darkTheme: Theme = {
    mode: 'dark',
    background: 'rgb(15, 23, 42)',
    canvas: '#0f172a',
    text: {
        primary: '#ffffff',
        secondary: '#94a3b8',
        outline: '#1e293b'
    },
    nodes: {
        getLayerColor: (level: number, maxLevel: number) => {
            // Dark mode: Cyan to Blue
            const t = 0.2 + 0.8 * (level / (maxLevel || 1));
            const r = Math.floor(20 * (1 - t) + 59 * t);
            const g = Math.floor(184 * (1 - t) + 130 * t);
            const b = Math.floor(166 * (1 - t) + 246 * t);
            return `rgb(${r}, ${g}, ${b})`;
        },
        activeColor: '#10b981',
        candidateColor: '#10b981'
    },
    edges: {
        permanent: '#475569',
        permanent_opacity: 0.3,
        vertical: '#334155',
        vertical_opacity: 0.5,
        persistent: '#3b82f6',
        selected: '#10b981',
        considered: '#f59e0b',
        rejected: '#ef4444'
    },
    planes: {
        getColor: (level: number, maxLevel: number) => {
            const intensity = 0.3 + 0.4 * (level / (maxLevel || 1));
            const gray = Math.floor(intensity * 100);
            return `rgb(${gray}, ${gray}, ${Math.floor(gray * 1.2)})`;
        },
        opacity: 0.12,
        gridPrimary: '#334155',
        gridSecondary: '#1e293b'
    },
    lighting: {
        ambient: 0.5,
        directional1: 1.0,
        directional2: 0.6,
        hemisphereTop: '#64748b',
        hemisphereBottom: '#1e293b'
    }
};

export const getTheme = (mode: ThemeMode): Theme => {
    return mode === 'light' ? lightTheme : darkTheme;
};
