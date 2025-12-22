# HNSW 3D Visualization Frontend

This is the frontend client for the Simple-HNSW project, built to visualize the Hierarchical Navigable Small World (HNSW) graph in interactive 3D.

## 🛠️ Tech Stack

- **Framework**: [React](https://react.dev/) (TypeScript)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **3D Engine**: [React Three Fiber](https://docs.pmnd.rs/react-three-fiber) (Three.js wrapper)
- **Animation**: [React Spring](https://www.react-spring.dev/) (Smooth 3D transitions)
- **Styling**: [TailwindCSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)

## 🚀 Getting Started

### Prerequisites
- Node.js 16+
- npm

### Installation

```bash
cd web_app/frontend
npm install
```

### Running Locally

```bash
npm run dev
```
The app will open at `http://localhost:5173`.
*Note: Ensure the Backend API is running on port 8000.*

## 📂 Project Structure

- **`src/components/HNSWScene.tsx`**: The core 3D scene containing the Graph, Nodes, and Edges visualization. Handles the logic for rendering layers, animations, and camera controls.
- **`src/components/Controls.tsx`**: The floating UI panel for user inputs (Insert, Search settings, Animation speed).
- **`src/App.tsx`**: Main application logic, including state management for the graph, animation sequencing, and API integration.
- **`src/theme.ts`**: Configuration for Light/Dark mode themes.

## 🎨 UI Features

- **Dark/Light Mode**: Toggle between a professional dark "Data Lab" theme and a clean light theme.
- **Glassmorphism**: Modern translucent UI elements.
- **Interactive Graph**:
    - **Hover**: Highlight nodes and their connections.
    - **Animation**: Step-by-step visualization of the search and insert processes.
    - **Layers**: Clear visualization of the HNSW hierarchy using 3D planes.

## 📦 Building for Production

```bash
npm run build
```
The output will be in the `dist` folder, ready for deployment.
