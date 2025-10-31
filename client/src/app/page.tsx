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

  return (
    
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* <SplashCursor/> */}
      
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="relative">
            <div className="absolute top-0 right-0">
              <ModeToggle />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4 h-full">
              HNSW Image Search Engine
            </h1>
            <p className="text-gray-600 dark:text-gray-400 text-lg">
              Search for images using natural language or upload an image to find similar ones
            </p>
          </div>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSearch} className="mb-12">
          <div className="flex flex-col gap-6 max-w-4xl mx-auto">
            {/* Search Type Toggle */}
            <div className="flex justify-center">
              <div className={`bg-white dark:bg-gray-800 rounded-full p-1 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 ${searchType === 'text' || searchType === 'image' ? 'search-toggle-active' : ''}`}>
                <button
                  type="button"
                  onClick={() => handleSearchTypeChange('text')}
                  className={`search-toggle-button px-6 hover:cursor-pointer py-3 rounded-full font-semibold transition-all duration-300 ${searchType === 'text'
                      ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-md transform scale-105'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                >
                  🔍 Text Search
                </button>
                <button
                  type="button"
                  onClick={() => handleSearchTypeChange('image')}
                  className={`search-toggle-button px-6 hover:cursor-pointer py-3 rounded-full font-semibold transition-all duration-300 ${searchType === 'image'
                      ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-md transform scale-105'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                >
                  🖼️ Image Search
                </button>
              </div>
            </div>

            {/* Search Input */}
            <div key={animationKey} className={`${
              animationState === 'exit' ? 'search-content-exit' : 
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
                    className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 hover:cursor-pointer"
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
                      className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 hover:cursor-pointer"
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
                        className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all ease-in-out duration-200 transform hover:scale-105 flex items-center gap-2 hover:cursor-pointer"
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
              {/* <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm text-blue-800 dark:text-blue-300">
                  <strong>Note:</strong> Make sure the Flask backend is running on port 5000. Both text and image searches use CLIP embeddings for accurate results.
                </p>
              </div> */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
