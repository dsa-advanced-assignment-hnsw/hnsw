import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const api = {
    initRandom: (max_elements: number, m: number, ef_construction: number, dim: number, init_count: number) =>
        axios.post(`${API_URL}/init_random`, { max_elements, m, ef_construction, dim, init_count }),

    insert: (vector: number[]) =>
        axios.post(`${API_URL}/insert`, { vector }),

    search: (vector: number[], k: number, ef?: number) =>
        axios.post(`${API_URL}/search`, { vector, k, ef }),

    getState: () =>
        axios.get(`${API_URL}/state`)
};
