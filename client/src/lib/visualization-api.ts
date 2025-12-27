const API_URL = process.env.NEXT_PUBLIC_VIS_API_URL || 'https://huynguyen6906-visualize-hnsw.hf.space/api';

export const visualizationApi = {
    initRandom: async (max_elements: number, m: number, ef_construction: number, dim: number, init_count: number) => {
        const response = await fetch(`${API_URL}/init_random`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_elements, m, ef_construction, dim, init_count })
        });
        return response.json();
    },

    insert: async (vector: number[]) => {
        const response = await fetch(`${API_URL}/insert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vector })
        });
        return response.json();
    },

    search: async (vector: number[], k: number, ef?: number) => {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vector, k, ef })
        });
        return response.json();
    },

    getState: async () => {
        const response = await fetch(`${API_URL}/state`);
        const data = await response.json();
        // Return a sanitized object that matches GraphState interface
        return {
            nodes: Array.isArray(data.nodes) ? data.nodes : [],
            edges: Array.isArray(data.edges) ? data.edges : [],
            global_entry_point: data.global_entry_point ?? null
        };
    }
};
