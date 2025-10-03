# Quick Start Guide

Get your HNSW Image Search Engine running in 5 minutes! 🚀

## Prerequisites Check

Before starting, make sure you have:

- ✅ Python 3.8 or higher: `python3 --version`
- ✅ Node.js 18 or higher: `node --version`
- ✅ yarn or npm: `yarn --version` or `npm --version`
- ✅ The `images_embeds.h5` file in the `backend/` directory

## Option 1: Automated Setup (Recommended)

### Step 1: Start Backend (Terminal 1)

```bash
# Make the script executable (first time only)
chmod +x start-backend.sh

# Run the backend
./start-backend.sh
```

The backend will start at `http://localhost:5000`

### Step 2: Start Frontend (Terminal 2)

```bash
# Make the script executable (first time only)
chmod +x start-frontend.sh

# Run the frontend
./start-frontend.sh
```

The frontend will start at `http://localhost:3000`

### Step 3: Open in Browser

Visit [http://localhost:3000](http://localhost:3000) and start searching! 🎉

## Option 2: Manual Setup

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start server
python server.py
```

### Frontend Setup

```bash
# 1. Open new terminal and navigate to client
cd client

# 2. Install dependencies
yarn install
# or: npm install

# 3. Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:5000" > .env.local

# 4. Start development server
yarn dev
# or: npm run dev
```

## Verify Everything Works

1. **Backend Health Check:**
   ```bash
   curl http://localhost:5000/health
   ```
   
   Expected response:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "device": "cuda" or "cpu"
   }
   ```

2. **Test Search:**
   We use the following to test if the local API runs:
   ```bash
   curl -X POST http://localhost:5000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "dog", "k": 5}'
   ```
   If we deploy frontend to Vercel and use Ngrok to connect it to backend:
   ```bash
   curl -X POST https://johana-versional-unhomiletically.ngrok-free.dev/search \
     -H "Content-Type: application/json" \
     -d '{"query": "dog", "k": 5}'
   ```

3. **Open Frontend:**
   - Visit [http://localhost:3000](http://localhost:3000)
   - Enter a search query like "beach" or "mountain"
   - See results with similarity scores

## Example Searches

Try these queries to test the system:
- 🐕 "dog playing in park"
- 🏖️ "beach sunset"
- 🏔️ "mountain landscape"
- 🌃 "city at night"
- 🐱 "cat sleeping"

## Troubleshooting

### Backend Issues

**Problem: ModuleNotFoundError**
```bash
# Solution: Activate virtual environment and reinstall
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

**Problem: Port 5000 already in use**
```bash
# Solution: Find and kill the process
lsof -ti:5000 | xargs kill -9
# Or use a different port in server.py
```

**Problem: images_embeds.h5 not found**
```bash
# Solution: Make sure the file is in backend directory
ls -la backend/images_embeds.h5
```

### Frontend Issues

**Problem: Cannot connect to backend**
- Check backend is running: `curl http://localhost:5000/health`
- Verify `.env.local` exists with correct URL
- Check browser console for CORS errors

**Problem: Blank page or errors**
```bash
# Solution: Clean install
cd client
rm -rf node_modules .next
yarn install
yarn dev
```

## Next Steps

### For Development:
- Read [backend/README.md](backend/README.md) for API documentation
- Read [client/README.md](client/README.md) for UI customization
- Modify search parameters in `client/src/app/page.tsx`

### For Production:
- Follow [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guides
- Deploy backend to Railway/Render
- Deploy frontend to Vercel

## Quick Commands Reference

```bash
# Backend
cd backend
source venv/bin/activate  # Activate venv
python server.py          # Start server
deactivate               # Deactivate venv

# Frontend
cd client
yarn dev                 # Start dev server
yarn build              # Build for production
yarn start              # Start production server

# Both (from project root)
./start-backend.sh      # Start backend
./start-frontend.sh     # Start frontend
```

## Getting Help

- 📖 Check the main [README.md](README.md)
- 🚀 Read the [DEPLOYMENT.md](DEPLOYMENT.md) guide
- 💻 Review backend API docs in [backend/README.md](backend/README.md)
- 🎨 UI customization in [client/README.md](client/README.md)

---

**Happy Searching! 🔍✨** 