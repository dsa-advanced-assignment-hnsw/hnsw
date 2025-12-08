'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { ModeToggle } from '@/components/ui/mode-toogle';
// import SplashCursor from '@/components/SplashCursor';

interface SearchResult {
  path: string;
  score: number;
  image_data?: string | null; // Optional: prefetched image data from server
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
    if (hasFetched) return; // Prevent duplicate fetches

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
  }, [path]); // Only depend on path, not on functions

  const handleRetry = () => {
    if (retryCount < 3) { // Limit retries to prevent infinite loops
      setRetryCount(prev => prev + 1);
      setHasFetched(false); // Reset fetch flag to allow retry
      setLoading(true);
      // Trigger re-fetch by calling loadImageData directly
      loadImageData(path).then(data => {
        if (data) {
          setImageSrc(data);
        } else {
          handleImageError(index);
        }
        setLoading(false);
      });
    }
  };

  if (imageErrors.has(index)) {
    return (
      <div className="relative aspect-square bg-gray-200 dark:bg-gray-700">
        <div className="w-full h-full flex flex-col items-center justify-center p-4">
          <svg className="w-16 h-16 text-gray-400 dark:text-gray-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-xs text-center text-gray-600 dark:text-gray-400 font-mono break-all">
            {path}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative aspect-square bg-gray-200 dark:bg-gray-700">
      {loading ? (
        <div className="w-full h-full flex items-center justify-center">
          <svg className="animate-spin h-8 w-8 text-gray-400" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 714 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
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
        <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 text-xs p-2">
          <div className="mb-2 text-center">Failed to load image</div>
          <div className="mb-2 text-center break-all">{path}</div>
          {retryCount < 3 && (
            <button
              onClick={handleRetry}
              className="px-2 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors"
            >
              Retry ({retryCount}/3)
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
      if (data) {
        setImageSrc(data);
      } else {
        setError(true);
      }
      setLoading(false);
    };
    fetchImage();
  }, [path]); // Only depend on path, not on loadImageData function

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.target as HTMLImageElement;
    onImageLoad({
      width: img.naturalWidth,
      height: img.naturalHeight
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-100 dark:bg-gray-700">
        <svg className="animate-spin h-12 w-12 text-gray-400" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      </div>
    );
  }

  if (error || !imageSrc) {
    return (
      <div className="flex flex-col items-center justify-center h-96 bg-gray-100 dark:bg-gray-700 p-8">
        <svg className="w-16 h-16 text-gray-400 dark:text-gray-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <p className="text-gray-600 dark:text-gray-400 text-center">Failed to load image</p>
        <p className="text-xs text-gray-500 dark:text-gray-500 mt-2 break-all">{path}</p>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center bg-gray-100 dark:bg-gray-700 h-auto">
      <Image
        src={imageSrc}
        alt="Selected image"
        className="w-full object-contain"
        onLoad={handleImageLoad}
        width={800}
        height={600}
      />
    </div>
  );
}

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
  const [imageRetryCount, setImageRetryCount] = useState<Map<string, number>>(new Map()); // Track retries per URL
  const [selectedImage, setSelectedImage] = useState<{ result: SearchResult; index: number } | null>(null);
  const [modalImageInfo, setModalImageInfo] = useState<{ width: number; height: number } | null>(null);
  const [downloadingImage, setDownloadingImage] = useState(false);
  const [isModalClosing, setIsModalClosing] = useState(false);
  const [searchType, setSearchType] = useState<'text' | 'image'>('text');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [uploadedImagePreview, setUploadedImagePreview] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<{ type: 'text' | 'image', value: string }>({ type: 'text', value: '' });
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [animationKey, setAnimationKey] = useState(0);
  // const [displayType, setDisplayType] = useState<'text' | 'image'>('text');
  const [animationState, setAnimationState] = useState<'enter' | 'exit' | 'hidden'>('enter');

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
  const [uploadedMedicalImagePreview, setUploadedMedicalImagePreview] = useState<string | null>(null);
  const [medicalLastQuery, setMedicalLastQuery] = useState<{ type: 'text' | 'image', value: string }>({ type: 'text', value: '' });
  const [medicalCurrentPage, setMedicalCurrentPage] = useState(1);
  const [medicalImageErrors, setMedicalImageErrors] = useState<Set<number>>(new Set());
  const [medicalImageDataCache, setMedicalImageDataCache] = useState<Map<string, string>>(new Map());
  const [selectedMedicalImage, setSelectedMedicalImage] = useState<{ result: SearchResult; index: number } | null>(null);
  const [medicalModalImageInfo, setMedicalModalImageInfo] = useState<{ width: number; height: number } | null>(null);
  const [downloadingMedicalImage, setDownloadingMedicalImage] = useState(false);

  const IMAGES_PER_PAGE = 20;

  const validateK = (value: number): boolean => {
    if (isNaN(value) || value !== Math.floor(value)) {
      setKError('Please enter a valid integer');
      return false;
    }
    if (value <= 0) {
      setKError('Number of results must be positive');
      return false;
    }
    setKError('');
    return true;
  };

  const handleKChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value === '') {
      setK(20);
      setKError('');
      return;
    }

    const numValue = Number(value);
    setK(numValue);
    validateK(numValue);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate k before searching
    if (!validateK(k)) {
      setError(kError);
      return;
    }

    if (searchType === 'text') {
      if (!query.trim()) {
        setError('Please enter a search query');
        return;
      }
      await performTextSearch();
    } else {
      if (!uploadedImage) {
        setError('Please upload an image to search');
        return;
      }
      await performImageSearch();
    }
  };

  const performTextSearch = async () => {
    setLoading(true);
    setError('');
    setSearched(true);
    setCurrentPage(1); // Reset to first page on new search
    setImageErrors(new Set()); // Reset image errors on new search
    setImageRetryCount(new Map()); // Reset retry counts on new search

    try {
      // const apiUrl = 'http://localhost:5000';
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true', // Skip Ngrok warning page
        },
        body: JSON.stringify({ query, k }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
      setLastQuery({ type: 'text', value: query });
    } catch (err) {
      // setError('Failed to search images. Please make sure the backend server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const performImageSearch = async () => {
    if (!uploadedImage) return;

    setLoading(true);
    setError('');
    setSearched(true);
    setCurrentPage(1); // Reset to first page on new search
    setImageErrors(new Set()); // Reset image errors on new search
    setImageRetryCount(new Map()); // Reset retry counts on new search

    try {
      // const apiUrl = 'http://localhost:5000';
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
      const formData = new FormData();
      formData.append('image', uploadedImage);
      formData.append('k', k.toString());

      const response = await fetch(`${apiUrl}/search/image`, {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true', // Skip Ngrok warning page
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Image search failed');
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
      setLastQuery({ type: 'image', value: uploadedImage.name });
    } catch (err) {
      setError('Failed to search with image. Please make sure the backend server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
      if (!allowedTypes.includes(file.type)) {
        setError('Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, or WEBP files.');
        return;
      }

      // Validate file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File too large. Please upload an image smaller than 10MB.');
        return;
      }

      setUploadedImage(file);
      setError('');

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearUploadedImage = () => {
    setUploadedImage(null);
    setUploadedImagePreview(null);
    // Reset file input
    const fileInput = document.getElementById('image-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleSearchTypeChange = (newType: 'text' | 'image') => {
    if (newType === searchType || isTransitioning) return;

    setIsTransitioning(true);
    setAnimationState('exit');

    // Wait for exit animation to complete
    setTimeout(() => {
      setSearchType(newType);
      // setDisplayType(newType);
      setAnimationKey(prev => prev + 1);

      // Clear any previous errors when switching
      setError('');

      // Reset upload if switching away from image search
      if (newType === 'text') {
        clearUploadedImage();
      }

      // Start enter animation
      setAnimationState('enter');

      setTimeout(() => {
        setIsTransitioning(false);
      }, 50);
    }, 200); // Wait for exit animation duration
  };

  const handleImageError = (index: number) => {
    setImageErrors(prev => new Set(prev).add(index));
  };

  // Card image loader
  const loadImageData = async (path: string, prefetchedData?: string | null, retryAttempt: number = 0) => {
    // If image was prefetched by server, use it directly
    if (prefetchedData !== undefined) {
      if (prefetchedData) {
        // Cache the prefetched data
        setImageDataCache(prev => new Map(prev).set(path, prefetchedData));
        return prefetchedData;
      } else {
        // Server tried to fetch but failed
        return null;
      }
    }

    // Check cache first
    if (imageDataCache.has(path)) {
      return imageDataCache.get(path)!;
    }

    // Check if we've exceeded max retries for this URL
    const currentRetries = imageRetryCount.get(path) || 0;
    if (currentRetries >= 3) {
      console.log(`Max retries (3) reached for ${path.substring(0, 80)}...`);
      return null;
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

      // Check if path is a URL (for v2 backend) or local path (for v1 backend)
      const isUrl = path.startsWith('http://') || path.startsWith('https://');
      const endpoint = isUrl
        ? `${apiUrl}/image-proxy?url=${encodeURIComponent(path.trim())}`
        : `${apiUrl}/image/${path}`;

      const response = await fetch(endpoint, {
        headers: {
          'ngrok-skip-browser-warning': 'true', // Skip Ngrok warning page
        },
      });

      if (!response.ok) {
        // Increment retry count
        setImageRetryCount(prev => new Map(prev).set(path, currentRetries + 1));
        return null;
      }

      const data: ImageData = await response.json();

      // Cache the image data
      setImageDataCache(prev => new Map(prev).set(path, data.image_data));

      // Reset retry count on success
      setImageRetryCount(prev => {
        const newMap = new Map(prev);
        newMap.delete(path);
        return newMap;
      });

      return data.image_data;
    } catch (err) {
      console.error(`Error loading image ${path.substring(0, 80)}...:`, err);
      // Increment retry count on error
      setImageRetryCount(prev => new Map(prev).set(path, currentRetries + 1));
      return null;
    }
  };

  // Modal image loader (ensure it also updates cache)
  const handleModalImageLoad = async (path: string) => {
    // Reuse the same loadImageData function for consistency
    return loadImageData(path);
  };

  const downloadImage = async (path: string) => {
    setDownloadingImage(true);
    try {
      const imageData = await loadImageData(path);
      if (imageData) {
        // Create a download link
        const link = document.createElement('a');
        link.href = imageData;
        link.download = path.split('/').pop() || 'image.jpg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (err) {
      console.error('Error downloading image:', err);
    } finally {
      setDownloadingImage(false);
    }
  };

  const handleCloseModal = () => {
    setIsModalClosing(true);
    setTimeout(() => {
      setSelectedImage(null);
      setIsModalClosing(false);
      setModalImageInfo(null);
    }, 300); // Match the animation duration
  };

  // Paper search functions
  const PAPERS_PER_PAGE = 20;

  const validatePaperK = (value: number): boolean => {
    if (isNaN(value) || value !== Math.floor(value)) {
      setPaperKError('Please enter a valid integer');
      return false;
    }
    if (value <= 0) {
      setPaperKError('Number of results must be positive');
      return false;
    }
    setPaperKError('');
    return true;
  };

  const handlePaperKChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value === '') {
      setPaperK(20);
      setPaperKError('');
      return;
    }

    const numValue = Number(value);
    setPaperK(numValue);
    validatePaperK(numValue);
  };

  const handlePaperSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validatePaperK(paperK)) {
      setPaperError(paperKError);
      return;
    }

    if (paperSearchType === 'text') {
      if (!paperQuery.trim()) {
        setPaperError('Please enter a search query');
        return;
      }
      await performPaperTextSearch();
    } else {
      if (!uploadedFile) {
        setPaperError('Please upload a file to search');
        return;
      }
      await performPaperFileSearch();
    }
  };

  // Fix URLs ending with .pd to .pdf
  const fixPaperUrl = (url: string): string => {
    if (url.endsWith('.pd')) {
      return url.slice(0, -3) + '.pdf';
    }
    return url;
  };

  const performPaperTextSearch = async () => {
    setPaperLoading(true);
    setPaperError('');
    setPaperSearched(true);
    setPaperCurrentPage(1);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_PAPER_API_URL || 'http://localhost:5001';
      const response = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({ query: paperQuery, k: paperK }),
      });

      if (!response.ok) {
        throw new Error('Paper search failed');
      }

      const data: PaperSearchResponse = await response.json();
      // Fix URLs ending with .pd
      const fixedResults = data.results.map(result => ({
        ...result,
        url: fixPaperUrl(result.url)
      }));
      setPaperResults(fixedResults);
      setPaperLastQuery({ type: 'text', value: paperQuery });
    } catch (err) {
      setPaperError('Failed to search papers. Please make sure the backend server is running.');
      console.error(err);
    } finally {
      setPaperLoading(false);
    }
  };

  const performPaperFileSearch = async () => {
    if (!uploadedFile) return;

    setPaperLoading(true);
    setPaperError('');
    setPaperSearched(true);
    setPaperCurrentPage(1);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_PAPER_API_URL || 'http://localhost:5001';
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('k', paperK.toString());

      const response = await fetch(`${apiUrl}/search/file`, {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('File-based paper search failed');
      }

      const data: PaperSearchResponse = await response.json();
      // Fix URLs ending with .pd
      const fixedResults = data.results.map(result => ({
        ...result,
        url: fixPaperUrl(result.url)
      }));
      setPaperResults(fixedResults);
      setPaperLastQuery({ type: 'file', value: uploadedFile.name });
    } catch (err) {
      setPaperError('Failed to search with file. Please make sure the backend server is running.');
      console.error(err);
    } finally {
      setPaperLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const allowedTypes = ['text/plain', 'application/pdf', 'text/markdown'];
      const allowedExtensions = ['.txt', '.pdf', '.md'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

      if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        setPaperError('Invalid file type. Please upload .txt, .pdf, or .md files.');
        return;
      }

      if (file.size > 10 * 1024 * 1024) {
        setPaperError('File too large. Please upload a file smaller than 10MB.');
        return;
      }

      setUploadedFile(file);
      setPaperError('');
    }
  };

  const clearUploadedFile = () => {
    setUploadedFile(null);
    const fileInput = document.getElementById('file-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handlePaperSearchTypeChange = (newType: 'text' | 'file') => {
    if (newType === paperSearchType) return;
    setPaperSearchType(newType);
    setPaperError('');
    if (newType === 'text') {
      clearUploadedFile();
    }
  };

  // Medical search handlers
  const handleMedicalSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!medicalQuery.trim()) {
      setMedicalError('Please enter a medical query');
      return;
    }
    if (!validateMedicalK(medicalK)) return;

    setMedicalLoading(true);
    setMedicalError('');
    setMedicalSearched(false);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_MEDICAL_API_URL || 'http://localhost:5002';
      const response = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: medicalQuery, k: medicalK }),
      });

      if (!response.ok) throw new Error('Search failed');
      
      const data: SearchResponse = await response.json();
      setMedicalResults(data.results);
      setMedicalSearched(true);
      setMedicalLastQuery({ type: 'text', value: medicalQuery });
      setMedicalCurrentPage(1);
    } catch (err) {
      setMedicalError('Failed to search. Please ensure the medical server is running on port 5002.');
    } finally {
      setMedicalLoading(false);
    }
  };

  const handleMedicalImageSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadedMedicalImage) {
      setMedicalError('Please upload an X-ray image');
      return;
    }
    if (!validateMedicalK(medicalK)) return;

    setMedicalLoading(true);
    setMedicalError('');

    try {
      const apiUrl = process.env.NEXT_PUBLIC_MEDICAL_API_URL || 'http://localhost:5002';
      const formData = new FormData();
      formData.append('image', uploadedMedicalImage);
      formData.append('k', medicalK.toString());

      const response = await fetch(`${apiUrl}/search/image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Image search failed');
      
      const data: SearchResponse = await response.json();
      setMedicalResults(data.results);
      setMedicalSearched(true);
      setMedicalLastQuery({ type: 'image', value: uploadedMedicalImage.name });
      setMedicalCurrentPage(1);
    } catch (err) {
      setMedicalError('Failed to search by image. Please check the server.');
    } finally {
      setMedicalLoading(false);
    }
  };
  
  const validateMedicalK = (value: number): boolean => {
    if (isNaN(value) || value !== Math.floor(value)) {
      setMedicalKError('Please enter a valid integer');
      return false;
    }
    if (value <= 0) {
      setMedicalKError('Number of results must be positive');
      return false;
    }
    setMedicalKError('');
    return true;
  };

  const loadMedicalImageData = async (path: string): Promise<string | null> => {
    if (medicalImageDataCache.has(path)) {
      return medicalImageDataCache.get(path)!;
    }

    try {
      // Check if path is already a full URL (Cloudinary URL)
      const isUrl = path.startsWith('http://') || path.startsWith('https://');
      
      if (isUrl) {
        // For Cloudinary URLs, fetch the image and convert to data URI
        const response = await fetch(path, {
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        });
        
        if (!response.ok) {
          console.error(`Failed to load medical image from URL: ${path.substring(0, 80)}...`, response.status);
          return null;
        }
        
        const blob = await response.blob();
        const reader = new FileReader();
        
        return new Promise((resolve) => {
          reader.onloadend = () => {
            const dataUri = reader.result as string;
            setMedicalImageDataCache(prev => new Map(prev).set(path, dataUri));
            resolve(dataUri);
          };
          reader.onerror = () => {
            console.error(`Error converting medical image to data URI: ${path.substring(0, 80)}...`);
            resolve(null);
          };
          reader.readAsDataURL(blob);
        });
      } else {
        // For relative paths, use the API endpoint
        const apiUrl = process.env.NEXT_PUBLIC_MEDICAL_API_URL || 'http://localhost:5002';
        
        // Use query parameter to avoid issues with encoded slashes in path
        const response = await fetch(`${apiUrl}/image?path=${encodeURIComponent(path)}`, {
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        });
        
        if (!response.ok) {
          console.error(`Failed to load medical image: ${path.substring(0, 80)}...`, response.status);
          return null;
        }
        
        const data: ImageData = await response.json();
        setMedicalImageDataCache(prev => new Map(prev).set(path, data.image_data));
        return data.image_data;
      }
    } catch (err) {
      console.error(`Error loading medical image ${path.substring(0, 80)}...:`, err);
      return null;
    }
  };

  const downloadMedicalImage = async (path: string) => {
    setDownloadingMedicalImage(true);
    try {
      const imageData = await loadMedicalImageData(path);
      if (imageData) {
        // Create a download link
        const link = document.createElement('a');
        link.href = imageData;
        link.download = path.split('/').pop() || 'xray-image.jpg';
        document.body.appendChild(link);
        link.click(); 
        document.body.removeChild(link);
      }
    } catch (err) {
      console.error('Error downloading medical image:', err);
    } finally {
      setDownloadingMedicalImage(false);
    }
  };

  const handleSearchModeChange = (newMode: 'image' | 'paper' | 'medical') => {
    if (newMode === searchMode) return;
    setSearchMode(newMode);
    // Reset errors when switching modes
    setError('');
    setPaperError('');
    setMedicalError('');
  };

  return (

    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* <SplashCursor/> */}

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="relative">
            <div className="absolute top-0 right-0">
              <ModeToggle />
            </div>
            <h1 className="text-5xl font-bold text-gray-900 dark:text-white font-wide mb-4 h-full">
              HNSW Search Engine
            </h1>
            <p className="text-gray-600 dark:text-gray-400 text-lg">
              {searchMode === 'image'
                ? 'Search for images using natural language or upload an image to find similar ones'
                : searchMode === 'paper'
                ? 'Search for academic papers using keywords or upload a document to find similar papers'
                : 'Search for bone fracture X-rays using medical terminology or upload an X-ray image'}
            </p>
          </div>
        </div>

        {/* Main Mode Toggle - Image vs Paper vs Medical */}
        <div className="flex justify-center mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-full p-1 shadow-lg border border-gray-200 dark:border-gray-700">
            <button
              type="button"
              onClick={() => handleSearchModeChange('image')}
              className={`px-8 py-3 rounded-full font-semibold transition-all duration-300 ${
                searchMode === 'image'
                  ? 'bg-blue-600 text-white shadow-md transform scale-105'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              🖼️ Image Search Engine
            </button>
            <button
              type="button"
              onClick={() => handleSearchModeChange('paper')}
              className={`px-8 py-3 rounded-full font-semibold transition-all duration-300 ${
                searchMode === 'paper'
                  ? 'bg-blue-600 text-white shadow-md transform scale-105'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              📄 Paper Search Engine
            </button>
            <button
              type="button"
              onClick={() => handleSearchModeChange('medical')}
              className={`px-8 py-3 rounded-full font-semibold transition-all duration-300 ${
                searchMode === 'medical'
                  ? 'bg-blue-600 text-white shadow-md transform scale-105'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              🦴 Fracture Search Engine
            </button>
          </div>
        </div>

        {/* Conditional Rendering based on searchMode */}
        {searchMode === 'image' ? (
          /* Image Search UI */
          <>
          <form onSubmit={handleSearch} className="mb-12">
          <div className="flex flex-col gap-6 max-w-4xl mx-auto">
            {/* Search Type Toggle */}
            <div className="flex justify-center">
              <div className={`bg-white dark:bg-gray-800 rounded-full p-1 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 ${searchType === 'text' || searchType === 'image' ? 'search-toggle-active' : ''}`}>
                <button
                  type="button"
                  onClick={() => handleSearchTypeChange('text')}
                  className={`search-toggle-button px-6 hover:cursor-pointer py-3 rounded-full font-semibold transition-all duration-300 ${searchType === 'text'
                    ? 'bg-blue-600 text-white shadow-md transform scale-105'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                >
                  🔍 Text Search
                </button>
                <button
                  type="button"
                  onClick={() => handleSearchTypeChange('image')}
                  className={`search-toggle-button px-6 hover:cursor-pointer py-3 rounded-full font-semibold transition-all duration-300 ${searchType === 'image'
                    ? 'bg-blue-600 text-white shadow-md transform scale-105'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                >
                  🖼️ Image Search
                </button>
              </div>
            </div>

            {/* Search Input */}
            <div key={animationKey} className={`${animationState === 'exit' ? 'search-content-exit' :
                animationState === 'enter' ? 'search-content-enter' :
                  'search-content-hidden'
              }`}>
              {searchType === 'text' ? (
                <div className='flex gap-4 ' id='textSearch'>
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search for images... (e.g., 'beach sunset', 'mountain landscape', 'dog playing')"
                    className="flex-1 px-6 py-4 rounded-full border-2 border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 dark:focus:border-blue-400 transition-all duration-300 shadow-lg hover:shadow-xl focus:shadow-xl"
                  />
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-8 py-4 bg-blue-600 text-white rounded-full font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 hover:cursor-pointer"
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Searching...
                      </span>
                    ) : (
                      'Search'
                    )}
                  </button>
                </div>
              ) : (
                <div className="space-y-4" id='imageSearch'>
                  {/* Image Upload Area */}
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-2xl p-8 bg-white dark:bg-gray-800 transition-all duration-300 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-gray-50 dark:hover:bg-gray-750">
                    {uploadedImagePreview ? (
                      <div className="flex flex-col items-center gap-4">
                        <div className="relative">
                          <Image
                            src={uploadedImagePreview}
                            alt="Uploaded image preview"
                            width={200}
                            height={200}
                            className="rounded-lg object-cover max-h-48 shadow-lg transition-all duration-300 hover:shadow-xl"
                            unoptimized
                          />
                          <button
                            type="button"
                            onClick={clearUploadedImage}
                            className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center hover:bg-red-600 transition-all duration-200 hover:cursor-pointer hover:scale-110 shadow-lg"
                          >
                            ×
                          </button>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 transition-colors duration-300">{uploadedImage?.name}</p>
                      </div>
                    ) : (
                      <div className="text-center">
                        <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500 transition-colors duration-300" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        <div className="mt-4">
                          <label htmlFor="image-upload" className="cursor-pointer">
                            <span className="text-blue-600 dark:text-blue-400 font-semibold hover:text-blue-700 dark:hover:text-blue-300 transition-colors duration-300">Upload an image</span>
                            <span className="text-gray-600 dark:text-gray-400 transition-colors duration-300"> or drag and drop</span>
                          </label>
                          <input
                            id="image-upload"
                            type="file"
                            className="hidden"
                            accept="image/*"
                            onChange={handleImageUpload}
                          />
                        </div>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 transition-colors duration-300">PNG, JPG, JPEG, GIF, BMP, WEBP up to 10MB</p>
                      </div>
                    )}
                  </div>

                  {/* Search Button for Image */}
                  <div className="flex justify-center">
                    <button
                      type="submit"
                      disabled={loading || !uploadedImage}
                      className="px-8 py-4 bg-blue-600 text-white rounded-full font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 hover:cursor-pointer"
                    >
                      {loading ? (
                        <span className="flex items-center gap-2">
                          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Searching...
                        </span>
                      ) : (
                        'Find Similar Images'
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Results Count Input */}
            <div className="flex items-center gap-4 justify-center">
              <label htmlFor="k-input" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Number of results:
              </label>
              <div className="relative">
                <input
                  id="k-input"
                  type="number"
                  value={k}
                  onChange={handleKChange}
                  className={`w-32 px-4 py-2 rounded-lg border-2 ${kError ? 'border-red-500 dark:border-red-400' : 'border-gray-200 dark:border-gray-700'} bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-center focus:outline-none focus:border-blue-500 dark:focus:border-blue-400 transition-all duration-200 shadow-sm hover:shadow-md focus:shadow-lg hover:border-gray-300 dark:hover:border-gray-600 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none`}
                />
                {kError && (
                  <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-red-500 dark:text-red-400 whitespace-nowrap">
                    {kError}
                  </div>
                )}
                {!kError && (
                  <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
                    (positive integers only)
                  </div>
                )}
              </div>
            </div>


          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="max-w-3xl mx-auto mb-8 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {/* Results */}
        {searched && !loading && results.length > 0 && (
          <div className="mb-8">
            <h2 className="text-2xl font-semibold mb-6 text-gray-900 dark:text-white">
              Found {results.length} result{results.length !== 1 ? 's' : ''} for{' '}
              {lastQuery.type === 'text' ? (
                <span>&quot;{lastQuery.value}&quot;</span>
              ) : (
                <span>image &quot;{lastQuery.value}&quot;</span>
              )}
            </h2>

            {/* Pagination Controls - Top */}
            {results.length > IMAGES_PER_PAGE && (
              <div className="flex justify-center items-center gap-4 mb-6">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                >
                  Previous
                </button>
                <span className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                  Page {currentPage} of {Math.ceil(results.length / IMAGES_PER_PAGE)}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(Math.ceil(results.length / IMAGES_PER_PAGE), p + 1))}
                  disabled={currentPage === Math.ceil(results.length / IMAGES_PER_PAGE)}
                  className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                >
                  Next
                </button>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {results
                .slice((currentPage - 1) * IMAGES_PER_PAGE, currentPage * IMAGES_PER_PAGE)
                .map((result, index) => {
                  const actualIndex = (currentPage - 1) * IMAGES_PER_PAGE + index;
                  return (
                    <div
                      key={actualIndex}
                      className="bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition-all transform hover:scale-105 border border-gray-200 dark:border-gray-700 cursor-pointer"
                      onClick={() => setSelectedImage({ result, index: actualIndex })}
                    >
                      <ImageDisplay
                        path={result.path}
                        index={actualIndex}
                        prefetchedData={result.image_data}
                        loadImageData={loadImageData}
                        imageErrors={imageErrors}
                        handleImageError={handleImageError}
                      />
                    </div>
                  );
                })}
            </div>

            {/* Pagination Controls - Bottom */}
            {results.length > IMAGES_PER_PAGE && (
              <div className="flex justify-center items-center gap-4 mt-6">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                >
                  Previous
                </button>
                <span className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                  Page {currentPage} of {Math.ceil(results.length / IMAGES_PER_PAGE)}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(Math.ceil(results.length / IMAGES_PER_PAGE), p + 1))}
                  disabled={currentPage === Math.ceil(results.length / IMAGES_PER_PAGE)}
                  className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                >
                  Next
                </button>
              </div>
            )}

            {/* Image Modal */}
            {selectedImage && (
              <div
                className={`fixed inset-0 backdrop-blur-sm bg-opacity-100 flex items-center justify-center z-50 p-4 ${isModalClosing ? 'modal-backdrop-exit' : 'modal-backdrop-enter'
                  }`}
                onClick={handleCloseModal}
              >
                <div
                  className={`relative max-w-5xl max-h-[90vh] bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-2xl ${isModalClosing ? 'modal-content-exit' : 'modal-content-enter'
                    }`}
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Close button */}
                  <button
                    onClick={handleCloseModal}
                    className="absolute top-4 right-4 z-10 bg-black bg-opacity-50 text-white rounded-full p-2 hover:bg-opacity-75 transition-all ease-in-out duration-200 hover:scale-110 hover:cursor-pointer"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>

                  <ModalImageDisplay
                    path={selectedImage.result.path}
                    loadImageData={handleModalImageLoad}
                    onImageLoad={setModalImageInfo}
                  />

                  {/* Image info and controls */}
                  <div className="p-6 bg-white dark:bg-gray-800">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                      <div className="flex-1 min-w-0">
                        <a
                          href={selectedImage.result.path}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-lg font-semibold text-blue-600 dark:text-blue-400 hover:underline mb-2 block truncate"
                          title={selectedImage.result.path}
                        >
                          {selectedImage.result.path}
                        </a>
                        <div className="flex flex-wrap gap-4 text-sm text-gray-600 dark:text-gray-400">
                          {modalImageInfo && (
                            <span>Resolution: {modalImageInfo.width} × {modalImageInfo.height}</span>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={() => downloadImage(selectedImage.result.path)}
                        disabled={downloadingImage}
                        className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all ease-in-out duration-200 transform hover:scale-105 flex items-center gap-2 hover:cursor-pointer"
                      >
                        {downloadingImage ? (
                          <>
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Downloading...
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Download
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* No Results */}
        {searched && !loading && results.length === 0 && !error && (
          <div className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400 text-lg">
              No results found for{' '}
              {lastQuery.type === 'text' ? (
                <span>&quot;{lastQuery.value}&quot;</span>
              ) : (
                <span>the uploaded image</span>
              )}
              . Try a different {lastQuery.type === 'text' ? 'search term' : 'image'}.
            </p>
          </div>
        )}

        {/* Instructions */}
        {!searched && (
          <div className="max-w-3xl mx-auto mt-16">
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl border border-gray-200 dark:border-gray-700">
              <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                How to use
              </h3>
              <ul className="space-y-3 text-gray-600 dark:text-gray-400">
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 dark:text-blue-400 font-bold">1.</span>
                  <span>Choose your search method: text query or image upload</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 dark:text-blue-400 font-bold">2.</span>
                  <span>For text search: enter descriptive text like &quot;beach sunset&quot; or &quot;mountain landscape&quot;</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 dark:text-blue-400 font-bold">3.</span>
                  <span>For image search: upload an image (PNG, JPG, GIF, etc.) to find similar images</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 dark:text-blue-400 font-bold">4.</span>
                  <span>Set the number of results you want (1-100, default: 20)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 dark:text-blue-400 font-bold">5.</span>
                  <span>Click search and browse similar images ranked by similarity score</span>
                </li>
              </ul>
            </div>
          </div>
            )}
          </>
        ) : searchMode === 'paper' ? (
          /* Paper Search UI */
          <>
            <form onSubmit={handlePaperSearch} className="mb-12">
              <div className="flex flex-col gap-6 max-w-4xl mx-auto">
                {/* Paper Search Type Toggle */}
                <div className="flex justify-center">
                  <div className="bg-white dark:bg-gray-800 rounded-full p-1 shadow-lg border border-gray-200 dark:border-gray-700">
                    <button
                      type="button"
                      onClick={() => handlePaperSearchTypeChange('text')}
                      className={`px-6 py-3 rounded-full font-semibold transition-all duration-300 ${
                        paperSearchType === 'text'
                          ? 'bg-blue-600 text-white shadow-md transform scale-105'
                          : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      🔍 Text Search
                    </button>
                    <button
                      type="button"
                      onClick={() => handlePaperSearchTypeChange('file')}
                      className={`px-6 py-3 rounded-full font-semibold transition-all duration-300 ${
                        paperSearchType === 'file'
                          ? 'bg-blue-600 text-white shadow-md transform scale-105'
                          : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      📁 File Search
                    </button>
                  </div>
                </div>

                {/* Paper Search Input */}
                {paperSearchType === 'text' ? (
                  <div className="flex gap-4">
                    <input
                      type="text"
                      value={paperQuery}
                      onChange={(e) => setPaperQuery(e.target.value)}
                      placeholder="Search for papers... (e.g., 'machine learning', 'deep neural networks', 'quantum computing')"
                      className="flex-1 px-6 py-4 rounded-full border-2 border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 dark:focus:border-blue-400 transition-all duration-300 shadow-lg hover:shadow-xl focus:shadow-xl"
                    />
                    <button
                      type="submit"
                      disabled={paperLoading}
                      className="px-8 py-4 bg-blue-600 text-white rounded-full font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
                    >
                      {paperLoading ? (
                        <span className="flex items-center gap-2">
                          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Searching...
                        </span>
                      ) : (
                        'Search'
                      )}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* File Upload Area */}
                    <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-2xl p-8 bg-white dark:bg-gray-800 transition-all duration-300 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-gray-50 dark:hover:bg-gray-750">
                      {uploadedFile ? (
                        <div className="flex flex-col items-center gap-4">
                          <div className="bg-blue-100 dark:bg-blue-900/30 p-4 rounded-lg">
                            <svg className="w-12 h-12 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">{uploadedFile.name}</p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">{(uploadedFile.size / 1024).toFixed(2)} KB</p>
                          <button
                            type="button"
                            onClick={clearUploadedFile}
                            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all duration-200 shadow-lg hover:shadow-xl"
                          >
                            Remove File
                          </button>
                        </div>
                      ) : (
                        <div className="text-center">
                          <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                          <div className="mt-4">
                            <label htmlFor="file-upload" className="cursor-pointer">
                              <span className="text-blue-600 dark:text-blue-400 font-semibold hover:text-blue-700 dark:hover:text-blue-300">Upload a document</span>
                              <span className="text-gray-600 dark:text-gray-400"> or drag and drop</span>
                            </label>
                            <input
                              id="file-upload"
                              type="file"
                              className="hidden"
                              accept=".txt,.pdf,.md"
                              onChange={handleFileUpload}
                            />
                          </div>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">.txt, .pdf, .md up to 10MB</p>
                        </div>
                      )}
                    </div>

                    {/* Search Button for File */}
                    <div className="flex justify-center">
                      <button
                        type="submit"
                        disabled={paperLoading || !uploadedFile}
                        className="px-8 py-4 bg-blue-600 text-white rounded-full font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
                      >
                        {paperLoading ? (
                          <span className="flex items-center gap-2">
                            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 714 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Searching...
                          </span>
                        ) : (
                          'Find Similar Papers'
                        )}
                      </button>
                    </div>
                  </div>
                )}

                {/* Results Count Input */}
                <div className="flex items-center gap-4 justify-center">
                  <label htmlFor="paper-k-input" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Number of results:
                  </label>
                  <div className="relative">
                    <input
                      id="paper-k-input"
                      type="number"
                      value={paperK}
                      onChange={handlePaperKChange}
                      className={`w-32 px-4 py-2 rounded-lg border-2 ${
                        paperKError ? 'border-red-500 dark:border-red-400' : 'border-gray-200 dark:border-gray-700'
                      } bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-center focus:outline-none focus:border-blue-500 dark:focus:border-blue-400 transition-all duration-200 shadow-sm hover:shadow-md focus:shadow-lg [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none`}
                    />
                    {paperKError && (
                      <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-red-500 dark:text-red-400 whitespace-nowrap">
                        {paperKError}
                      </div>
                    )}
                    {!paperKError && (
                      <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
                        (positive integers only)
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </form>

            {/* Error Message */}
            {paperError && (
              <div className="max-w-3xl mx-auto mb-8 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
                <p className="font-semibold">Error:</p>
                <p>{paperError}</p>
              </div>
            )}

            {/* Paper Results */}
            {paperSearched && !paperLoading && paperResults.length > 0 && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold mb-6 text-gray-900 dark:text-white">
                  Found {paperResults.length} paper{paperResults.length !== 1 ? 's' : ''} for{' '}
                  {paperLastQuery.type === 'text' ? (
                    <span>&quot;{paperLastQuery.value}&quot;</span>
                  ) : (
                    <span>file &quot;{paperLastQuery.value}&quot;</span>
                  )}
                </h2>

                {/* Pagination Controls - Top */}
                {paperResults.length > PAPERS_PER_PAGE && (
                  <div className="flex justify-center items-center gap-4 mb-6">
                    <button
                      onClick={() => setPaperCurrentPage(p => Math.max(1, p - 1))}
                      disabled={paperCurrentPage === 1}
                      className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                    >
                      Previous
                    </button>
                    <span className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                      Page {paperCurrentPage} of {Math.ceil(paperResults.length / PAPERS_PER_PAGE)}
                    </span>
                    <button
                      onClick={() => setPaperCurrentPage(p => Math.min(Math.ceil(paperResults.length / PAPERS_PER_PAGE), p + 1))}
                      disabled={paperCurrentPage === Math.ceil(paperResults.length / PAPERS_PER_PAGE)}
                      className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                    >
                      Next
                    </button>
                  </div>
                )}

                {/* Paper Results List */}
                <div className="space-y-4">
                  {paperResults
                    .slice((paperCurrentPage - 1) * PAPERS_PER_PAGE, paperCurrentPage * PAPERS_PER_PAGE)
                    .map((result, index) => {
                      const actualIndex = (paperCurrentPage - 1) * PAPERS_PER_PAGE + index;
                      return (
                        <div
                          key={actualIndex}
                          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-2xl transition-all transform hover:scale-[1.02] border border-gray-200 dark:border-gray-700"
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-3 mb-2">
                                <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                  #{actualIndex + 1}
                                </span>
                                <div className="flex-1">
                                  <a
                                    href={result.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-lg font-semibold text-blue-600 dark:text-blue-400 hover:underline block truncate"
                                    title={result.url}
                                  >
                                    {result.url}
                                  </a>
                                </div>
                              </div>
                              <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                                <div className="flex items-center gap-2">
                                  <span className="font-medium">Similarity Score:</span>
                                  <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400 rounded-full font-semibold">
                                    {(result.similarity * 100).toFixed(2)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                            <a
                              href={result.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-all duration-200 transform hover:scale-105 flex items-center gap-2 whitespace-nowrap"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                              View Paper
                            </a>
                          </div>
                        </div>
                      );
                    })}
                </div>

                {/* Pagination Controls - Bottom */}
                {paperResults.length > PAPERS_PER_PAGE && (
                  <div className="flex justify-center items-center gap-4 mt-6">
                    <button
                      onClick={() => setPaperCurrentPage(p => Math.max(1, p - 1))}
                      disabled={paperCurrentPage === 1}
                      className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                    >
                      Previous
                    </button>
                    <span className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                      Page {paperCurrentPage} of {Math.ceil(paperResults.length / PAPERS_PER_PAGE)}
                    </span>
                    <button
                      onClick={() => setPaperCurrentPage(p => Math.min(Math.ceil(paperResults.length / PAPERS_PER_PAGE), p + 1))}
                      disabled={paperCurrentPage === Math.ceil(paperResults.length / PAPERS_PER_PAGE)}
                      className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* No Results */}
            {paperSearched && !paperLoading && paperResults.length === 0 && !paperError && (
              <div className="text-center py-12">
                <p className="text-gray-500 dark:text-gray-400 text-lg">
                  No results found for{' '}
                  {paperLastQuery.type === 'text' ? (
                    <span>&quot;{paperLastQuery.value}&quot;</span>
                  ) : (
                    <span>the uploaded file</span>
                  )}
                  . Try a different {paperLastQuery.type === 'text' ? 'search term' : 'file'}.
                </p>
              </div>
            )}

            {/* Instructions */}
            {!paperSearched && (
              <div className="max-w-3xl mx-auto mt-16">
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl border border-gray-200 dark:border-gray-700">
                  <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                    How to use
                  </h3>
                  <ul className="space-y-3 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-3">
                      <span className="text-blue-600 dark:text-blue-400 font-bold">1.</span>
                      <span>Choose your search method: text query or file upload</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="text-blue-600 dark:text-blue-400 font-bold">2.</span>
                      <span>For text search: enter keywords like &quot;machine learning&quot; or &quot;quantum computing&quot;</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="text-blue-600 dark:text-blue-400 font-bold">3.</span>
                      <span>For file search: upload a document (.txt, .pdf, .md) to find similar papers</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="text-blue-600 dark:text-blue-400 font-bold">4.</span>
                      <span>Set the number of results you want (positive integer, default: 20)</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="text-blue-600 dark:text-blue-400 font-bold">5.</span>
                      <span>Click search and browse similar papers ranked by similarity score</span>
                    </li>
                  </ul>
                  <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <p className="text-sm text-blue-800 dark:text-blue-300">
                      <strong>Note:</strong> This search engine uses HNSW algorithm to search through 1 million arXiv papers using Sentence Transformer embeddings.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : searchMode === 'medical' ? (
          /* Medical Search UI */
          <>
            <form onSubmit={medicalSearchType === 'text' ? handleMedicalSearch : handleMedicalImageSearch} className="mb-12">
              <div className="flex flex-col gap-6 max-w-4xl mx-auto">
                {/* Medical Search Type Toggle */}
                <div className="flex justify-center">
                  <div className="bg-white dark:bg-gray-800 rounded-full p-1 shadow-lg border border-gray-200 dark:border-gray-700">
                    <button
                      type="button"
                      onClick={() => setMedicalSearchType('text')}
                      className={`px-6 py-3 rounded-full font-semibold transition-all duration-300 ${
                        medicalSearchType === 'text'
                          ? 'bg-blue-600 text-white shadow-md transform scale-105'
                          : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      🔍 Text Query
                    </button>
                    <button
                      type="button"
                      onClick={() => setMedicalSearchType('image')}
                      className={`px-6 py-3 rounded-full font-semibold transition-all duration-300 ${
                        medicalSearchType === 'image'
                          ? 'bg-blue-600 text-white shadow-md transform scale-105'
                          : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      🩻 X-ray Image
                    </button>
                  </div>
                </div>

                {medicalSearchType === 'text' ? (
                  <div className="flex flex-col gap-4">
                    <input
                      type="text"
                      value={medicalQuery}
                      onChange={(e) => setMedicalQuery(e.target.value)}
                      placeholder="e.g., distal radius fracture, broken leg bone, femur fracture..."
                      className="flex-1 px-6 py-4 text-lg rounded-xl border-2 border-gray-300 dark:border-gray-600 focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
                    />
                    <div className="flex gap-4 items-center">
                      <label className="text-gray-700 dark:text-gray-300 font-medium">Results (k):</label>
                      <input
                        type="number"
                        value={medicalK}
                        onChange={(e) => setMedicalK(parseInt(e.target.value) || 20)}
                        min="1"
                        max="100"
                        className="w-24 px-4 py-2 rounded-lg border-2 border-gray-300 dark:border-gray-600 focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                      />
                      <button
                        type="submit"
                        disabled={medicalLoading}
                        className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {medicalLoading ? 'Searching...' : 'Search X-rays'}
                      </button>
                    </div>
                    {medicalKError && <p className="text-red-500 text-sm">{medicalKError}</p>}
                  </div>
                ) : (
                  <div className="flex flex-col gap-4">
                    <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-8 text-center">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            setUploadedMedicalImage(file);
                            setUploadedMedicalImagePreview(URL.createObjectURL(file));
                          }
                        }}
                        className="hidden"
                        id="medical-image-upload"
                      />
                      <label htmlFor="medical-image-upload" className="cursor-pointer">
                        {uploadedMedicalImagePreview ? (
                          <div className="space-y-4">
                            <img src={uploadedMedicalImagePreview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
                            <p className="text-sm text-gray-600 dark:text-gray-400">{uploadedMedicalImage?.name}</p>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <svg className="w-16 h-16 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <p className="text-gray-600 dark:text-gray-400">Click to upload X-ray image</p>
                          </div>
                        )}
                      </label>
                    </div>
                    <div className="flex gap-4 items-center justify-center">
                      <label className="text-gray-700 dark:text-gray-300 font-medium">Results (k):</label>
                      <input
                        type="number"
                        value={medicalK}
                        onChange={(e) => setMedicalK(parseInt(e.target.value) || 20)}
                        min="1"
                        max="100"
                        className="w-24 px-4 py-2 rounded-lg border-2 border-gray-300 dark:border-gray-600 focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                      />
                      <button
                        type="submit"
                        disabled={medicalLoading || !uploadedMedicalImage}
                        className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {medicalLoading ? 'Searching...' : 'Find Similar X-rays'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </form>

            {medicalError && (
              <div className="mb-8 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-red-600 dark:text-red-400">{medicalError}</p>
              </div>
            )}

            {medicalSearched && medicalResults.length > 0 && (
              <div className="mb-8">
                <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
                  Search Results ({medicalResults.length} X-rays found)
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Query: <span className="font-semibold">{medicalLastQuery.value}</span>
                  {medicalLastQuery.type === 'image' && ' (X-ray image search)'}
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                  {medicalResults
                    .slice((medicalCurrentPage - 1) * IMAGES_PER_PAGE, medicalCurrentPage * IMAGES_PER_PAGE)
                    .map((result, index) => (
                      <div
                        key={index}
                        onClick={() => setSelectedMedicalImage({ result, index })}
                        className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-all duration-300 cursor-pointer transform hover:scale-105"
                      >
                        <ImageDisplay
                          path={result.path}
                          index={index}
                          loadImageData={loadMedicalImageData}
                          imageErrors={medicalImageErrors}
                          handleImageError={(idx) => setMedicalImageErrors(prev => new Set(prev).add(idx))}
                        />
                      </div>
                    ))}
                </div>

                {medicalResults.length > IMAGES_PER_PAGE && (
                  <div className="mt-8 flex justify-center gap-2">
                    <button
                      onClick={() => setMedicalCurrentPage(prev => Math.max(1, prev - 1))}
                      disabled={medicalCurrentPage === 1}
                      className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg disabled:opacity-50"
                    >
                      Previous
                    </button>
                    <span className="px-4 py-2 text-gray-700 dark:text-gray-300">
                      Page {medicalCurrentPage} of {Math.ceil(medicalResults.length / IMAGES_PER_PAGE)}
                    </span>
                    <button
                      onClick={() => setMedicalCurrentPage(prev => Math.min(Math.ceil(medicalResults.length / IMAGES_PER_PAGE), prev + 1))}
                      disabled={medicalCurrentPage >= Math.ceil(medicalResults.length / IMAGES_PER_PAGE)}
                      className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg disabled:opacity-50"
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Medical Image Modal */}
            {selectedMedicalImage && (
              <div className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4">
                <div className="relative bg-white dark:bg-gray-800 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
                  {/* Close button */}
                  <button
                    onClick={() => setSelectedMedicalImage(null)}
                    className="sticky top-4 left-full z-10 bg-white dark:bg-gray-700 rounded-full p-2 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors shadow-lg ml-auto mr-4 mb-2"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>

                  <div className="px-4 pb-4">
                    <ModalImageDisplay
                      path={selectedMedicalImage.result.path}
                      loadImageData={loadMedicalImageData}
                      onImageLoad={setMedicalModalImageInfo}
                    />
                  </div>

                  {/* Image info and controls */}
                  <div className="p-6 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                      <div className="flex-1 min-w-0">
                        <p className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2 truncate" title={selectedMedicalImage.result.path}>
                          {selectedMedicalImage.result.path.split('/').pop() || 'X-ray Image'}
                        </p>
                        <div className="flex flex-wrap gap-4 text-sm text-gray-600 dark:text-gray-400">
                          {medicalModalImageInfo && (
                            <span>Resolution: {medicalModalImageInfo.width} × {medicalModalImageInfo.height}</span>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={() => downloadMedicalImage(selectedMedicalImage.result.path)}
                        disabled={downloadingMedicalImage}
                        className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all ease-in-out duration-200 transform hover:scale-105 flex items-center gap-2 hover:cursor-pointer whitespace-nowrap"
                      >
                        {downloadingMedicalImage ? (
                          <>
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Downloading...
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Download
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
