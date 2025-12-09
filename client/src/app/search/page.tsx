'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { ModeToggle } from '@/components/ui/mode-toogle';
import Link from 'next/link';
import { ArrowLeft, Search, Upload, FileText, Image as ImageIcon, Activity, X, Download, AlertCircle, ChevronLeft, ChevronRight, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// --- Interfaces ---
interface SearchResult {
  path: string;
  score: number;
  image_data?: string | null;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

interface PaperResult {
  url: string;
  similarity: number;
}

interface PaperSearchResponse {
  query?: string;
  filename?: string;
  k: number;
  results: PaperResult[];
}

interface ImageData {
  image_data: string;
  path: string;
}

// --- Components ---

function ImageDisplay({
  path,
  index,
  prefetchedData,
  loadImageData,
  imageErrors,
  handleImageError
}: {
  path: string;
  index: number;
  prefetchedData?: string | null;
  loadImageData: (path: string, prefetchedData?: string | null) => Promise<string | null>;
  imageErrors: Set<number>;
  handleImageError: (index: number) => void;
}) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [hasFetched, setHasFetched] = useState(false);

  const fetchImage = async () => {
    if (hasFetched) return;
    setLoading(true);
    setHasFetched(true);

    const data = await loadImageData(path, prefetchedData);
    if (data) {
      setImageSrc(data);
    } else {
      handleImageError(index);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchImage();
  }, [path]);

  const handleRetry = () => {
    if (retryCount < 3) {
      setRetryCount(prev => prev + 1);
      setHasFetched(false);
      setLoading(true);
      loadImageData(path).then(data => {
        if (data) setImageSrc(data);
        else handleImageError(index);
        setLoading(false);
      });
    }
  };

  if (imageErrors.has(index)) {
    return (
      <div className="w-full h-full bg-red-100 flex flex-col items-center justify-center p-4 border-b-4 border-black">
        <AlertCircle className="w-12 h-12 text-black mb-2" />
        <p className="text-xs font-mono font-bold text-center break-all">{path}</p>
      </div>
    );
  }

  return (
    <div className="relative aspect-square w-full h-full bg-white border-b-4 border-black">
      {loading ? (
        <div className="w-full h-full flex items-center justify-center bg-gray-100">
          <div className="w-10 h-10 border-4 border-black border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : imageSrc ? (
        <Image
          src={imageSrc}
          alt={`Result ${index + 1}`}
          fill
          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
          className="object-cover"
          unoptimized
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center bg-gray-200">
          <span className="font-bold mb-2">FAILED</span>
          {retryCount < 3 && (
            <button
              onClick={handleRetry}
              className="px-2 py-1 bg-black text-white text-xs font-bold hover:bg-blue-600"
            >
              RETRY ({retryCount}/3)
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function ModalImageDisplay({
  path,
  loadImageData,
  onImageLoad
}: {
  path: string;
  loadImageData: (path: string) => Promise<string | null>;
  onImageLoad: (info: { width: number; height: number }) => void;
}) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchImage = async () => {
      setLoading(true);
      setError(false);
      const data = await loadImageData(path);
      if (data) setImageSrc(data);
      else setError(true);
      setLoading(false);
    };
    fetchImage();
  }, [path]);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.target as HTMLImageElement;
    onImageLoad({
      width: img.naturalWidth,
      height: img.naturalHeight
    });
  };

  if (loading) {
    return (
      <div className="min-h-[400px] flex items-center justify-center">
        <div className="w-16 h-16 border-8 border-black border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error || !imageSrc) {
    return (
      <div className="min-h-[400px] flex flex-col items-center justify-center bg-red-100 border-4 border-black p-8">
        <AlertCircle className="w-16 h-16 mb-4" />
        <p className="font-black text-xl">FAILED TO LOAD IMAGE</p>
        <p className="font-mono mt-2 break-all bg-white px-2 border-2 border-black">{path}</p>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center bg-black/5 p-4 min-h-[400px]">
      <Image
        src={imageSrc}
        alt="Selected image"
        className="max-w-full max-h-[70vh] object-contain border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] bg-white"
        onLoad={handleImageLoad}
        width={1200}
        height={800}
      />
    </div>
  );
}


// --- Main Page Component ---

export default function Home() {
  // Main mode: 'image' or 'paper' or 'medical'
  const [searchMode, setSearchMode] = useState<'image' | 'paper' | 'medical'>('image');

  // Image search states
  const [query, setQuery] = useState('');
  const [k, setK] = useState(20);
  const [kError, setKError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searched, setSearched] = useState(false);
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());
  const [imageDataCache, setImageDataCache] = useState<Map<string, string>>(new Map());
  const [imageRetryCount, setImageRetryCount] = useState<Map<string, number>>(new Map());
  const [selectedImage, setSelectedImage] = useState<{ result: SearchResult; index: number } | null>(null);
  const [modalImageInfo, setModalImageInfo] = useState<{ width: number; height: number } | null>(null);
  const [downloadingImage, setDownloadingImage] = useState(false);
  const [searchType, setSearchType] = useState<'text' | 'image'>('text');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [uploadedImagePreview, setUploadedImagePreview] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<{ type: 'text' | 'image', value: string }>({ type: 'text', value: '' });

  // Paper search states
  const [paperQuery, setPaperQuery] = useState('');
  const [paperK, setPaperK] = useState(20);
  const [paperKError, setPaperKError] = useState('');
  const [paperResults, setPaperResults] = useState<PaperResult[]>([]);
  const [paperLoading, setPaperLoading] = useState(false);
  const [paperError, setPaperError] = useState('');
  const [paperSearched, setPaperSearched] = useState(false);
  const [paperSearchType, setPaperSearchType] = useState<'text' | 'file'>('text');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [paperLastQuery, setPaperLastQuery] = useState<{ type: 'text' | 'file', value: string }>({ type: 'text', value: '' });
  const [paperCurrentPage, setPaperCurrentPage] = useState(1);

  // Medical search states
  const [medicalQuery, setMedicalQuery] = useState('');
  const [medicalK, setMedicalK] = useState(20);
  const [medicalKError, setMedicalKError] = useState('');
  const [medicalResults, setMedicalResults] = useState<SearchResult[]>([]);
  const [medicalLoading, setMedicalLoading] = useState(false);
  const [medicalError, setMedicalError] = useState('');
  const [medicalSearched, setMedicalSearched] = useState(false);
  const [medicalSearchType, setMedicalSearchType] = useState<'text' | 'image'>('text');
  const [uploadedMedicalImage, setUploadedMedicalImage] = useState<File | null>(null);
  const [medicalImagePreview, setMedicalImagePreview] = useState<string | null>(null);
  const [medicalLastQuery, setMedicalLastQuery] = useState<{ type: 'text' | 'image', value: string }>({ type: 'text', value: '' });
  const [medicalCurrentPage, setMedicalCurrentPage] = useState(1);
  const [medicalImageDataCache, setMedicalImageDataCache] = useState<Map<string, string>>(new Map());
  const [medicalImageErrors, setMedicalImageErrors] = useState<Set<number>>(new Set());

  const IMAGES_PER_PAGE = 20;
  const PAPERS_PER_PAGE = 20;

  // --- Handlers ---

  const validateK = (value: number, setErrorFn: (s: string) => void) => {
    if (isNaN(value) || value !== Math.floor(value)) {
      setErrorFn('INTEGER REQUIRED');
      return false;
    }
    if (value <= 0) {
      setErrorFn('MUST BE > 0');
      return false;
    }
    setErrorFn('');
    return true;
  };

  const handleKChange = (e: React.ChangeEvent<HTMLInputElement>, setFn: (n: number) => void, setErrorFn: (s: string) => void) => {
    const val = e.target.value;
    if (val === '') {
      setFn(20);
      setErrorFn('');
      return;
    }
    const num = Number(val);
    setFn(num);
    validateK(num, setErrorFn);
  };

  // Image Search
  const performTextSearch = async () => {
    setLoading(true); setError(''); setSearched(true); setCurrentPage(1);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
      const res = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
        body: JSON.stringify({ query, k }),
      });
      if (!res.ok) throw new Error('Search failed');
      const data: SearchResponse = await res.json();
      setResults(data.results);
      setLastQuery({ type: 'text', value: query });
    } catch (err) { console.error(err); setError('SEARCH FAILED. CHECK BACKEND.'); }
    finally { setLoading(false); }
  };

  const performImageSearch = async () => {
    if (!uploadedImage) return;
    setLoading(true); setError(''); setSearched(true); setCurrentPage(1);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
      const fd = new FormData();
      fd.append('image', uploadedImage);
      fd.append('k', k.toString());
      const res = await fetch(`${apiUrl}/search/image`, {
        method: 'POST',
        headers: { 'ngrok-skip-browser-warning': 'true' },
        body: fd,
      });
      if (!res.ok) throw new Error('Image search failed');
      const data: SearchResponse = await res.json();
      setResults(data.results);
      setLastQuery({ type: 'image', value: uploadedImage.name });
    } catch (err) { console.error(err); setError('IMAGE UPLOAD FAILED.'); }
    finally { setLoading(false); }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateK(k, setKError)) { setError('INVALID COUNT'); return; }
    if (searchType === 'text') {
      if (!query.trim()) { setError('EMPTY QUERY'); return; }
      performTextSearch();
    } else {
      if (!uploadedImage) { setError('NO IMAGE'); return; }
      performImageSearch();
    }
  };

  // Paper Search
  const performPaperTextSearch = async () => {
    setPaperLoading(true); setPaperError(''); setPaperSearched(true); setPaperCurrentPage(1);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_PAPER_API_URL || 'http://localhost:5001';
      const res = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
        body: JSON.stringify({ query: paperQuery, k: paperK }),
      });
      if (!res.ok) throw new Error('Paper search failed');
      const data: PaperSearchResponse = await res.json();
      const fixed = data.results.map(r => ({ ...r, url: r.url.endsWith('.pd') ? r.url.replace('.pd', '.pdf') : r.url }));
      setPaperResults(fixed);
      setPaperLastQuery({ type: 'text', value: paperQuery });
    } catch (err) { console.error(err); setPaperError('PAPER SEARCH FAILED.'); }
    finally { setPaperLoading(false); }
  };

  const performPaperFileSearch = async () => {
    if (!uploadedFile) return;
    setPaperLoading(true); setPaperError(''); setPaperSearched(true); setPaperCurrentPage(1);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_PAPER_API_URL || 'http://localhost:5001';
      const fd = new FormData();
      fd.append('file', uploadedFile);
      fd.append('k', paperK.toString());
      const res = await fetch(`${apiUrl}/search/file`, {
        method: 'POST',
        headers: { 'ngrok-skip-browser-warning': 'true' },
        body: fd,
      });
      if (!res.ok) throw new Error('Paper file search failed');
      const data: PaperSearchResponse = await res.json();
      const fixed = data.results.map(r => ({ ...r, url: r.url.endsWith('.pd') ? r.url.replace('.pd', '.pdf') : r.url }));
      setPaperResults(fixed);
      setPaperLastQuery({ type: 'file', value: uploadedFile.name });
    } catch (err) { console.error(err); setPaperError('FILE SEARCH FAILED.'); }
    finally { setPaperLoading(false); }
  };

  const handlePaperSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateK(paperK, setPaperKError)) { setPaperError('INVALID COUNT'); return; }
    if (paperSearchType === 'text') {
      if (!paperQuery.trim()) { setPaperError('EMPTY QUERY'); return; }
      performPaperTextSearch();
    } else {
      if (!uploadedFile) { setPaperError('NO FILE'); return; }
      performPaperFileSearch();
    }
  };

  // Medical Search - Text
  const performMedicalTextSearch = async () => {
    setMedicalLoading(true); setMedicalError(''); setMedicalSearched(false);
    setMedicalImageErrors(new Set());
    try {
      const apiUrl = process.env.NEXT_PUBLIC_MEDICAL_API_URL || 'http://localhost:5002';
      const res = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: medicalQuery, k: medicalK }),
      });
      if (!res.ok) throw new Error('Medical search failed');
      const data = await res.json();
      setMedicalResults(data.results);
      setMedicalSearched(true);
      setMedicalLastQuery({ type: 'text', value: medicalQuery });
      setMedicalCurrentPage(1);
    } catch (err) { setMedicalError('FAILED. CHECK BACKEND 5002.'); }
    finally { setMedicalLoading(false); }
  };

  // Medical Search - Image
  const performMedicalImageSearch = async () => {
    if (!uploadedMedicalImage) return;
    setMedicalLoading(true); setMedicalError(''); setMedicalSearched(false);
    setMedicalImageErrors(new Set());
    try {
      const apiUrl = process.env.NEXT_PUBLIC_MEDICAL_API_URL || 'http://localhost:5002';
      const fd = new FormData();
      fd.append('image', uploadedMedicalImage);
      fd.append('k', medicalK.toString());
      const res = await fetch(`${apiUrl}/search/image`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) throw new Error('Medical image search failed');
      const data = await res.json();
      setMedicalResults(data.results);
      setMedicalSearched(true);
      setMedicalLastQuery({ type: 'image', value: uploadedMedicalImage.name });
      setMedicalCurrentPage(1);
    } catch (err) { setMedicalError('IMAGE SEARCH FAILED. CHECK BACKEND 5002.'); }
    finally { setMedicalLoading(false); }
  };

  const handleMedicalSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateK(medicalK, setMedicalKError)) { setMedicalError('INVALID COUNT'); return; }
    if (medicalSearchType === 'text') {
      if (!medicalQuery.trim()) { setMedicalError('EMPTY QUERY'); return; }
      performMedicalTextSearch();
    } else {
      if (!uploadedMedicalImage) { setMedicalError('NO IMAGE'); return; }
      performMedicalImageSearch();
    }
  };

  // Image Loader Helpers (Reusing logic from original)
  const loadImageData = async (path: string, prefetchedData?: string | null) => {
    if (prefetchedData) return prefetchedData;
    if (imageDataCache.has(path)) {
      return imageDataCache.get(path)!;
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
      const isUrl = path.startsWith('http://') || path.startsWith('https://');

      // For any external URL, use Next.js API route to proxy (avoids CORS issues)
      if (isUrl) {
        const proxyUrl = `/api/image-proxy?url=${encodeURIComponent(path)}`;
        const res = await fetch(proxyUrl);
        if (!res.ok) return null;
        const blob = await res.blob();
        const dataUri = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.onerror = () => resolve('');
          reader.readAsDataURL(blob);
        });
        if (dataUri) {
          setImageDataCache(prev => new Map(prev).set(path, dataUri));
        }
        return dataUri;
      }

      // For local/static paths, use the backend API image endpoint
      const res = await fetch(`${apiUrl}/image/${path}`, { headers: { 'ngrok-skip-browser-warning': 'true' } });
      if (!res.ok) return null;
      const data: ImageData = await res.json();
      setImageDataCache(prev => new Map(prev).set(path, data.image_data));
      return data.image_data;
    } catch (error) {
      console.error('Error in loadImageData:', error);
      return null;
    }
  };

  const loadMedicalImageData = async (path: string, prefetchedData?: string | null) => {
    // Use prefetched data if available
    if (prefetchedData) {
      if (prefetchedData) {
        setMedicalImageDataCache(prev => new Map(prev).set(path, prefetchedData));
        return prefetchedData;
      }
      return null;
    }

    // Check cache
    if (medicalImageDataCache.has(path)) {
      return medicalImageDataCache.get(path)!;
    }

    try {
      const isUrl = path.startsWith('http://') || path.startsWith('https://');

      // For any external URL (including Cloudinary), use Next.js API route to proxy (avoids CORS)
      if (isUrl) {
        const proxyUrl = `/api/image-proxy?url=${encodeURIComponent(path)}`;
        const res = await fetch(proxyUrl);
        if (!res.ok) {
          console.error(`Failed to fetch medical image: ${path}`, res.status);
          return null;
        }
        const blob = await res.blob();
        const dataUri = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.onerror = () => {
            console.error(`Error converting medical image to data URI: ${path}`);
            resolve('');
          };
          reader.readAsDataURL(blob);
        });
        if (dataUri) {
          setMedicalImageDataCache(prev => new Map(prev).set(path, dataUri));
        }
        return dataUri;
      }

      // For local paths, use the medical API endpoint
      const apiUrl = process.env.NEXT_PUBLIC_MEDICAL_API_URL || 'http://localhost:5002';
      const res = await fetch(`${apiUrl}/image?path=${encodeURIComponent(path)}`);
      if (!res.ok) return null;
      const data: ImageData = await res.json();
      setMedicalImageDataCache(prev => new Map(prev).set(path, data.image_data));
      return data.image_data;
    } catch (error) {
      console.error(`Error loading medical image: ${path}`, error);
      return null;
    }
  };

  // Render Theme Helpers
  const getThemeColor = () => {
    if (searchMode === 'image') return 'bg-[#FF00D6]';
    if (searchMode === 'paper') return 'bg-[#00F0FF]';
    return 'bg-yellow-400';
  };

  const getBorderColor = () => 'border-black';

  return (
    <div className="min-h-screen bg-[#FFFDF5] text-black font-sans selection:bg-black selection:text-white">

      {/* --- Top App Bar --- */}
      <nav className="border-b-4 border-black bg-white sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <Link href="/" className="w-12 h-12 bg-black text-white flex items-center justify-center font-black text-xl hover:bg-blue-600 transition-colors border-2 border-transparent hover:border-black">
              <ArrowLeft />
            </Link>
            <h1 className="text-3xl font-black italic tracking-tighter uppercase hidden sm:block">
              HNSW <span className="text-blue-600">ENGINE</span>
            </h1>
          </div>
          <div className="flex gap-2">
            {/* <div className="font-mono text-sm font-bold border-2 border-black px-2 py-1 bg-[#FFFDF5] shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
              V2.0.0
            </div> */}
            {/* <ModeToggle /> */}
          </div>
        </div>
      </nav>  

      <main className="max-w-7xl mx-auto px-4 py-12">

        {/* --- Mode Selector --- */}
        <div className="flex flex-wrap justify-center gap-4 mb-16">
          {[
            { id: 'image', label: 'IMAGE SEARCH', icon: <ImageIcon className="w-5 h-5" />, color: 'bg-[#FF00D6]' },
            { id: 'paper', label: 'RESEARCH PAPERS', icon: <FileText className="w-5 h-5" />, color: 'bg-[#00F0FF]' },
            { id: 'medical', label: 'MED DIAGNOSTICS', icon: <Activity className="w-5 h-5" />, color: 'bg-yellow-400' }
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => { setSearchMode(mode.id as any); setError(''); setPaperError(''); setMedicalError(''); }}
              className={`
                 relative px-8 py-4 font-black text-xl flex items-center gap-3 border-4 border-black transition-all
                 ${searchMode === mode.id
                  ? 'bg-black text-white translate-x-[2px] translate-y-[2px] shadow-none'
                  : 'bg-white text-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:-translate-y-1 hover:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:bg-[#FFFDF5]'}
               `}
            >
              {mode.icon} {mode.label}
            </button>
          ))}
        </div>

        {/* --- Mode Content --- */}
        <div className={`border-4 border-black p-8 shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] bg-white relative ${searchMode === 'medical' ? 'pattern-diagonal-lines-sm' : ''}`}>

          {/* Decorative Badge */}
          <div className={`absolute -top-6 -left-6 ${getThemeColor()} border-4 border-black px-4 py-1 transform -rotate-3 font-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            {searchMode === 'image' ? 'CLIP ENABLED' : searchMode === 'paper' ? 'ARXIV INDEX' : 'HIPAA READY'}
          </div>

          {/* IMAGE SEARCH UI */}
          {searchMode === 'image' && (
            <div>
              <div className="flex justify-CENTER mb-8">
                <div className="inline-flex border-4 border-black bg-white">
                  <button
                    onClick={() => setSearchType('text')}
                    className={`px-6 py-2 font-bold uppercase transition-colors ${searchType === 'text' ? 'bg-black text-white' : 'hover:bg-gray-100'}`}
                  >
                    Text Query
                  </button>
                  <button
                    onClick={() => setSearchType('image')}
                    className={`px-6 py-2 font-bold uppercase transition-colors border-l-4 border-black ${searchType === 'image' ? 'bg-black text-white' : 'hover:bg-gray-100'}`}
                  >
                    Image Upload
                  </button>
                </div>
              </div>

              <form onSubmit={handleSearch} className="max-w-4xl mx-auto flex flex-col gap-6">
                <div className="flex flex-col md:flex-row gap-4">
                  {searchType === 'text' ? (
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      className="flex-1 border-4 border-black p-4 font-bold text-xl placeholder:text-gray-400 focus:outline-none focus:ring-4 ring-blue-400 bg-[#FFFDF5]"
                      placeholder="DESCRIBE IMAGE (E.G. 'CYBERPUNK CITY')..."
                    />
                  ) : (
                    <div className="flex-1 border-4 border-black border-dashed bg-gray-50 p-4 flex items-center justify-center relative cursor-pointer hover:bg-gray-100">
                      <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => {
                        if (e.target.files) {
                          setUploadedImage(e.target.files[0]);
                          const reader = new FileReader();
                          reader.onload = (ev) => setUploadedImagePreview(ev.target?.result as string);
                          reader.readAsDataURL(e.target.files[0]);
                        }
                      }} accept="image/*" />
                      {uploadedImagePreview ? (
                        <img src={uploadedImagePreview} alt="Preview" className="h-16 w-16 object-cover border-2 border-black" />
                      ) : (
                        <span className="font-bold flex items-center gap-2 text-gray-500"><Upload /> UPLOAD IMAGE</span>
                      )}
                    </div>
                  )}

                  <div className="w-32">
                    <input
                      type="number"
                      value={k}
                      onChange={(e) => handleKChange(e, setK, setKError)}
                      className="w-full h-full border-4 border-black p-4 font-bold text-center bg-white"
                    />
                  </div>

                  <button type="submit" disabled={loading} className="border-4 border-black bg-[#FF00D6] px-8 py-4 font-black text-white shadow-[4px_4px_0px_0px_#000] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_#000] active:translate-x-[4px] active:translate-y-[4px] active:shadow-none transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                    {loading ? 'Thinking...' : 'SEARCH'}
                  </button>
                </div>
                {error && <div className="bg-red-500 text-white font-bold p-2 text-center border-2 border-black shadow-[4px_4px_0px_0px_#000]">{error}</div>}
              </form>
            </div>
          )}

          {/* PAPER SEARCH UI */}
          {searchMode === 'paper' && (
            <div>
              <div className="flex justify-center mb-8">
                <div className="inline-flex border-4 border-black bg-white">
                  <button
                    onClick={() => setPaperSearchType('text')}
                    className={`px-6 py-2 font-bold uppercase transition-colors ${paperSearchType === 'text' ? 'bg-black text-white' : 'hover:bg-gray-100'}`}
                  >
                    Text Query
                  </button>
                  <button
                    onClick={() => setPaperSearchType('file')}
                    className={`px-6 py-2 font-bold uppercase transition-colors border-l-4 border-black ${paperSearchType === 'file' ? 'bg-black text-white' : 'hover:bg-gray-100'}`}
                  >
                    Similar Paper
                  </button>
                </div>
              </div>

              <form onSubmit={handlePaperSearch} className="max-w-4xl mx-auto flex flex-col gap-6">
                <div className="flex flex-col md:flex-row gap-4">
                  {paperSearchType === 'text' ? (
                    <input
                      type="text"
                      value={paperQuery}
                      onChange={(e) => setPaperQuery(e.target.value)}
                      className="flex-1 border-4 border-black p-4 font-bold text-xl placeholder:text-gray-400 focus:outline-none focus:ring-4 ring-cyan-400 bg-[#FFFDF5]"
                      placeholder="PAPER TOPIC (E.G. 'ATTENTION IS ALL YOU NEED')..."
                    />
                  ) : (
                    <div className="flex-1 border-4 border-black border-dashed bg-gray-50 p-4 flex items-center justify-center relative cursor-pointer hover:bg-gray-100">
                      <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => {
                        if (e.target.files) {
                          setUploadedFile(e.target.files[0]);
                        }
                      }} accept=".pdf,.txt" />
                      {uploadedFile ? (
                        <span className="font-bold flex items-center gap-2"><FileText className="w-6 h-6" /> {uploadedFile.name}</span>
                      ) : (
                        <span className="font-bold flex items-center gap-2 text-gray-500"><Upload /> UPLOAD PDF/TXT</span>
                      )}
                    </div>
                  )}

                  <div className="w-32">
                    <input
                      type="number"
                      value={paperK}
                      onChange={(e) => handleKChange(e, setPaperK, setPaperKError)}
                      className="w-full h-full border-4 border-black p-4 font-bold text-center bg-white"
                      placeholder="K"
                    />
                  </div>

                  <button type="submit" disabled={paperLoading} className="border-4 border-black bg-[#00F0FF] px-8 py-4 font-black text-black shadow-[4px_4px_0px_0px_#000] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_#000] transition-all disabled:opacity-50">
                    {paperLoading ? 'SCANNING...' : 'FIND'}
                  </button>
                </div>
                {paperError && <div className="bg-red-500 text-white font-bold p-2 text-center border-2 border-black shadow-[4px_4px_0px_0px_#000]">{paperError}</div>}
              </form>
            </div>
          )}

          {/* MEDICAL SEARCH UI */}
          {searchMode === 'medical' && (
            <div>
              <div className="flex justify-center mb-8">
                <div className="inline-flex border-4 border-black bg-white">
                  <button
                    onClick={() => setMedicalSearchType('text')}
                    className={`px-6 py-2 font-bold uppercase transition-colors ${medicalSearchType === 'text' ? 'bg-black text-white' : 'hover:bg-gray-100'}`}
                  >
                    Text Query
                  </button>
                  <button
                    onClick={() => setMedicalSearchType('image')}
                    className={`px-6 py-2 font-bold uppercase transition-colors border-l-4 border-black ${medicalSearchType === 'image' ? 'bg-black text-white' : 'hover:bg-gray-100'}`}
                  >
                    Similar Scan
                  </button>
                </div>
              </div>

              <form onSubmit={handleMedicalSearch} className="max-w-4xl mx-auto flex flex-col gap-6">
                <div className="flex flex-col md:flex-row gap-4">
                  {medicalSearchType === 'text' ? (
                    <input
                      type="text"
                      value={medicalQuery}
                      onChange={(e) => setMedicalQuery(e.target.value)}
                      className="flex-1 border-4 border-black p-4 font-bold text-xl placeholder:text-gray-400 focus:outline-none focus:ring-4 ring-yellow-400 bg-[#FFFDF5]"
                      placeholder="CONDITION (E.G. 'FRACTURE')..."
                    />
                  ) : (
                    <div className="flex-1 border-4 border-black border-dashed bg-gray-50 p-4 flex items-center justify-center relative cursor-pointer hover:bg-gray-100">
                      <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => {
                        if (e.target.files) {
                          setUploadedMedicalImage(e.target.files[0]);
                          const reader = new FileReader();
                          reader.onload = (ev) => setMedicalImagePreview(ev.target?.result as string);
                          reader.readAsDataURL(e.target.files[0]);
                        }
                      }} accept="image/*" />
                      {medicalImagePreview ? (
                        <img src={medicalImagePreview} alt="Preview" className="h-16 w-16 object-cover border-2 border-black" />
                      ) : (
                        <span className="font-bold flex items-center gap-2 text-gray-500"><Upload /> UPLOAD SCAN IMAGE</span>
                      )}
                    </div>
                  )}

                  <div className="w-32">
                    <input
                      type="number"
                      value={medicalK}
                      onChange={(e) => handleKChange(e, setMedicalK, setMedicalKError)}
                      className="w-full h-full border-4 border-black p-4 font-bold text-center bg-white"
                      placeholder="K"
                    />
                  </div>

                  <button type="submit" disabled={medicalLoading} className="border-4 border-black bg-yellow-400 px-8 py-4 font-black text-black shadow-[4px_4px_0px_0px_#000] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_#000] transition-all disabled:opacity-50">
                    {medicalLoading ? 'ANALYZING...' : 'DIAGNOSE'}
                  </button>
                </div>
                {medicalError && <div className="bg-red-500 text-white font-bold p-2 text-center border-2 border-black shadow-[4px_4px_0px_0px_#000]">{medicalError}</div>}
              </form>
            </div>
          )}
        </div>

        {/* --- RESULTS GRID --- */}

        {/* Image Results */}
        {searchMode === 'image' && searched && !loading && (
          <div className="mt-16">
            <h2 className="text-4xl font-black mb-8 uppercase flex items-center gap-4">
              RESULTS <span className="text-sm bg-black text-white px-2 py-1 font-mono">{results.length} FOUND</span>
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {results.slice((currentPage - 1) * IMAGES_PER_PAGE, currentPage * IMAGES_PER_PAGE).map((result, idx) => {
                const actualIdx = (currentPage - 1) * IMAGES_PER_PAGE + idx;
                return (
                  <div key={actualIdx} onClick={() => setSelectedImage({ result, index: actualIdx })} className="group border-4 border-black bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[4px] hover:translate-y-[4px] transition-all cursor-pointer">
                    <div className="aspect-square relative border-b-4 border-black overflow-hidden">
                      <ImageDisplay
                        path={result.path}
                        index={actualIdx}
                        prefetchedData={result.image_data}
                        loadImageData={loadImageData}
                        imageErrors={imageErrors}
                        handleImageError={(i) => setImageErrors(prev => new Set(prev).add(i))}
                      />
                    </div>
                    <div className="p-3 bg-white">
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-black text-lg">#{actualIdx + 1}</span>
                        <span className="font-mono text-xs bg-black text-white px-1">{(result.score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-gray-200 border border-black">
                        <div className="h-full bg-[#FF00D6]" style={{ width: `${result.score * 100}%` }}></div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Pagination */}
            {results.length > IMAGES_PER_PAGE && (
              <div className="flex justify-center gap-4 mt-8">
                <button onClick={() => setCurrentPage(c => Math.max(1, c - 1))} disabled={currentPage === 1} className="w-12 h-12 flex items-center justify-center border-4 border-black font-black hover:bg-black hover:text-white disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-black">
                  <ChevronLeft />
                </button>
                <span className="font-black text-xl flex items-center">{currentPage}</span>
                <button onClick={() => setCurrentPage(c => Math.min(Math.ceil(results.length / IMAGES_PER_PAGE), c + 1))} disabled={currentPage === Math.ceil(results.length / IMAGES_PER_PAGE)} className="w-12 h-12 flex items-center justify-center border-4 border-black font-black hover:bg-black hover:text-white disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-black">
                  <ChevronRight />
                </button>
              </div>
            )}
          </div>
        )}

        {/* Paper Results */}
        {searchMode === 'paper' && paperSearched && !paperLoading && (
          <div className="mt-16 max-w-5xl mx-auto">
            <h2 className="text-4xl font-black mb-8 uppercase">PAPERS FOUND</h2>
            <div className="flex flex-col gap-4">
              {paperResults.slice((paperCurrentPage - 1) * PAPERS_PER_PAGE, paperCurrentPage * PAPERS_PER_PAGE).map((result, idx) => (
                <div key={idx} className="border-4 border-black p-6 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[4px] hover:translate-y-[4px] transition-all flex items-start gap-4">
                  <div className="w-12 h-12 bg-[#00F0FF] border-4 border-black flex items-center justify-center font-black flex-shrink-0">
                    {idx + 1}
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <a href={result.url} target="_blank" className="font-bold text-xl hover:bg-yellow-200 truncate block border-b-2 border-transparent hover:border-black transition-all">
                      {result.url.split('/').pop()}
                    </a>
                    <div className="flex items-center gap-4 mt-2">
                      <span className="font-mono text-sm bg-gray-100 px-2 py-1 border border-black">PDF</span>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-sm">MATCH:</span>
                        <div className="w-32 h-3 border border-black bg-gray-100">
                          <div className="h-full bg-[#00F0FF]" style={{ width: `${result.similarity * 100}%` }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <a href={result.url} target="_blank" className="w-10 h-10 border-4 border-black flex items-center justify-center hover:bg-black hover:text-white transition-colors">
                    <Download className="w-4 h-4" />
                  </a>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Medical Results */}
        {searchMode === 'medical' && medicalSearched && !medicalLoading && (
          <div className="mt-16">
            <div className="bg-yellow-400 border-4 border-black p-4 mb-8 flex items-center gap-4 shadow-[8px_8px_0px_0px_#000]">
              <AlertCircle className="w-8 h-8" />
              <span className="font-black text-xl">CONFIDENTIAL MEDICAL RECORDS</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {medicalResults.slice((medicalCurrentPage - 1) * IMAGES_PER_PAGE, medicalCurrentPage * IMAGES_PER_PAGE).map((result, idx) => {
                const actualIdx = (medicalCurrentPage - 1) * IMAGES_PER_PAGE + idx;
                return (
                  <div key={actualIdx} onClick={() => setSelectedImage({ result, index: actualIdx })} className="group border-4 border-black bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[4px] hover:translate-y-[4px] transition-all cursor-pointer">
                    <div className="aspect-square relative border-b-4 border-black overflow-hidden">
                      <ImageDisplay
                        path={result.path}
                        index={actualIdx}
                        prefetchedData={result.image_data}
                        loadImageData={loadMedicalImageData}
                        imageErrors={medicalImageErrors}
                        handleImageError={(i) => setMedicalImageErrors(prev => new Set(prev).add(i))}
                      />
                    </div>
                    <div className="p-3 bg-white">
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-black text-lg">#{actualIdx + 1}</span>
                        <span className="font-mono text-xs bg-black text-white px-1">{(result.score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-gray-200 border border-black">
                        <div className="h-full bg-yellow-400" style={{ width: `${result.score * 100}%` }}></div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            {medicalResults.length > IMAGES_PER_PAGE && (
              <div className="flex justify-center gap-4 mt-8">
                <button onClick={() => setMedicalCurrentPage(c => Math.max(1, c - 1))} disabled={medicalCurrentPage === 1} className="w-12 h-12 flex items-center justify-center border-4 border-black font-black hover:bg-black hover:text-white disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-black">
                  <ChevronLeft />
                </button>
                <span className="font-black text-xl flex items-center">{medicalCurrentPage}</span>
                <button onClick={() => setMedicalCurrentPage(c => Math.min(Math.ceil(medicalResults.length / IMAGES_PER_PAGE), c + 1))} disabled={medicalCurrentPage === Math.ceil(medicalResults.length / IMAGES_PER_PAGE)} className="w-12 h-12 flex items-center justify-center border-4 border-black font-black hover:bg-black hover:text-white disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-black">
                  <ChevronRight />
                </button>
              </div>
            )}
          </div>
        )}

      </main>

      {/* --- Image Modal --- */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedImage(null)}
          >
            <div
              className="bg-white border-4 border-black p-2 max-w-6xl w-full shadow-[20px_20px_0px_0px_rgba(255,255,255,0.2)]"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex justify-between items-center mb-4 bg-black p-2">
                <span className="text-white font-mono font-bold truncate px-2">{selectedImage.result.path}</span>
                <button onClick={() => setSelectedImage(null)} className="bg-red-500 text-white w-8 h-8 font-black flex items-center justify-center border-2 border-white hover:scale-110 transition-transform">X</button>
              </div>

              <ModalImageDisplay
                path={selectedImage.result.path}
                loadImageData={loadImageData}
                onImageLoad={setModalImageInfo}
              />

              <div className="mt-4 flex gap-4">
                <a
                  href={selectedImage.result.path}
                  download
                  target="_blank"
                  className="flex-1 bg-black text-white font-black py-4 text-center hover:bg-blue-600 transition-colors uppercase border-4 border-transparent hover:border-black"
                >
                  Download High-Res
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
