# HNSW Image Search - Frontend

Modern Next.js frontend for the HNSW Image Search Engine.

## Features

- 🎨 Beautiful, responsive UI built with Next.js 15 and Tailwind CSS
- 🔍 Real-time image search with loading states
- 📊 Visual similarity scores with progress bars
- 🌓 Dark mode support
- ⚡ Optimized image loading with lazy loading
- 🎯 TypeScript for type safety

## Prerequisites

- Node.js 18+ or Bun
- Running backend server (see `../backend/README.md`)

## Installation

Install dependencies:
```bash
yarn install
# or
npm install
```

## Environment Variables

Create a `.env.local` file in the `client` folder:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

For production, update this to your deployed backend URL.

## Running Locally

Development server:
```bash
yarn dev
# or
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Building for Production

```bash
yarn build
yarn start
# or
npm run build
npm start
```

## Deploying to Vercel

### Method 1: Vercel CLI (Recommended)

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy from the client folder:
```bash
cd client
vercel
```

4. Set environment variable:
```bash
vercel env add NEXT_PUBLIC_API_URL
```
Enter your backend URL (e.g., `https://your-backend.railway.app`)

5. Redeploy with environment variable:
```bash
vercel --prod
```

### Method 2: Vercel Dashboard

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Add New Project"
3. Import your Git repository
4. Set root directory to `client`
5. Add environment variable:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: Your backend URL (e.g., `https://your-backend.railway.app`)
6. Click "Deploy"

### Method 3: Deploy Button

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/your-repo&root-directory=client&env=NEXT_PUBLIC_API_URL&envDescription=Backend%20API%20URL&envLink=https://github.com/yourusername/your-repo)

## Configuration

### Vercel Configuration

The `vercel.json` file is already configured:
- Build command: `npm run build`
- Output directory: `.next`
- Framework: Next.js

### API Integration

The frontend communicates with the backend through these endpoints:
- `POST /search` - Search for images
- `GET /image/:path` - Retrieve image
- `GET /health` - Health check

## Project Structure

```
client/
├── src/
│   └── app/
│       ├── page.tsx         # Main search page
│       ├── layout.tsx        # Root layout
│       └── globals.css       # Global styles
├── public/                   # Static assets
├── package.json              # Dependencies
├── vercel.json              # Vercel config
└── tsconfig.json            # TypeScript config
```

## Customization

### Styling
The app uses Tailwind CSS. Customize colors in `tailwind.config.js` or update the classes in `page.tsx`.

### Search Parameters
Modify the search request in `page.tsx`:
- `k`: Number of results (default: 20)
- API endpoint customization

### UI Components
All components are in `src/app/page.tsx`. Feel free to extract them into separate component files.

## Troubleshooting

### Backend Connection Issues
- Ensure backend is running on the specified URL
- Check CORS is enabled on backend
- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Check browser console for errors

### Vercel Deployment Issues
- Environment variables must start with `NEXT_PUBLIC_` to be exposed to browser
- Rebuild after adding/changing environment variables
- Check Vercel deployment logs for errors

### Images Not Loading
- Verify backend `/image/:path` endpoint is working
- Check image paths in the database
- Ensure backend has access to image files

## Performance

- Images are lazy-loaded
- API calls include error handling
- Optimized for Core Web Vitals
- Turbopack enabled for faster builds

## Support

For issues or questions:
1. Check backend is running: `GET http://your-backend/health`
2. Check browser console for errors
3. Verify environment variables
4. Review Vercel deployment logs
