export interface NodeData {
    id: number;
    layer: number;
    x: number;
    y: number;
    z: number;
    vector?: number[];
}

export interface EdgeData {
    u: number;
    v: number;
    layer: number;
}

export interface GraphState {
    nodes: NodeData[];
    edges: EdgeData[];
    global_entry_point?: number | null;
}

export interface BaseLog {
    event: string;
    [key: string]: any;
}

export type LogEvent =
    | { event: 'init_insert'; node_id: number; assigned_level: number; current_max_level: number; entry_point: number }
    | { event: 'node_insert'; level: number; id: number }
    | { event: 'zoom_in_layer'; level: number; ep: number }
    | { event: 'zoom_in_complete'; level: number; new_ep: number }
    | { event: 'construction_layer'; level: number; ep: number }
    | { event: 'candidates_found'; level: number; node_id: number; candidates: number[] }
    | { event: 'neighbors_selected'; level: number; node_id: number; neighbors: number[]; distances: number[] }
    | { event: 'add_connection'; level: number; edge: [number, number] }
    | { event: 'prune_connection'; level: number; edge: [number, number] }
    | { event: 'layer_transition_construction'; node_id: number; from_layer: number; to_layer: number; ep: number }
    | { event: 'init_search_layer'; layer: number; ep: number }
    | { event: 'visit_node'; current_node: number }
    | { event: 'consider_neighbor'; current_node: number; neighbor: number }
    | { event: 'accept_neighbor'; current_node: number; neighbor: number }
    | { event: 'reject_neighbor'; current_node: number; neighbor: number }
    | { event: 'reject_node'; current_node?: number; reject_node?: number; id?: number }
    | { event: 'reject_nodes'; reject_nodes: number[] }
    | { event: 'init_knn_search'; entry_point: number; max_level: number }
    | { event: 'layer_transition'; from_layer: number; to_layer: number; ep: number }
    | { event: 'current_node'; id: number }
    | { event: 'extend_candidate'; candidate: number }
    | { event: 'select_neighbor'; neighbor: number };
