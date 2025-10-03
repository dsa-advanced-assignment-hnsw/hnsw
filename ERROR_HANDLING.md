# Image Error Handling

## Overview

The application now gracefully handles cases where images cannot be loaded, displaying the file location instead of broken images.

## How It Works

### When Images Load Successfully ✅
```
┌──────────────────┐
│                  │
│   [IMAGE PREVIEW]│
│                  │
├──────────────────┤
│ Similarity: 89%  │
│ ████████████     │
│ filename.jpg     │
└──────────────────┘
```

### When Images Fail to Load ❌
```
┌──────────────────┐
│      📷          │
│                  │
│  ./images/123.jpg│
│                  │
├──────────────────┤
│ Similarity: 89%  │
│ ████████████     │
│ 123.jpg          │
└──────────────────┘
```

## Error Scenarios Handled

### 1. Image File Not Found
**Cause:** The image path exists in the database but the file is missing.

**Backend Response:**
```json
{
  "error": "Image not found",
  "path": "./images/missing.jpg",
  "message": "The image file does not exist at the specified location"
}
```

**Frontend Display:**
- Shows image icon (📷)
- Displays full file path in monospace font
- Still shows similarity score and progress bar

### 2. Cannot Read Image File
**Cause:** File exists but cannot be opened (permissions, corruption, etc.)

**Backend Response:**
```json
{
  "error": "Cannot read image",
  "path": "./images/corrupted.jpg",
  "message": "Failed to read image file: [error details]"
}
```

**Frontend Display:**
- Same fallback as above
- Path shown for debugging

### 3. Invalid File Type
**Cause:** File is not a supported image format.

**Backend Response:**
```json
{
  "error": "Invalid file type",
  "path": "./files/document.pdf",
  "message": "File is not a supported image format"
}
```

**Supported Formats:**
- `.jpg` / `.jpeg`
- `.png`
- `.gif`
- `.bmp`
- `.webp`

### 4. Network/CORS Errors
**Cause:** Backend unreachable or CORS issues.

**Frontend Behavior:**
- Triggers `onError` event on `<img>` tag
- Displays fallback with file path

## Visual Design

### Fallback UI Components:

1. **Icon:** 
   - SVG image icon (broken image symbol)
   - Gray color (#9CA3AF in light mode)
   - 64px × 64px size

2. **File Path:**
   - Monospace font for clarity
   - Word-break to prevent overflow
   - Centered alignment
   - Smaller text size (xs)

3. **Container:**
   - Same aspect ratio as successful images
   - Flexbox centering
   - Gray background matching loading state

## Implementation Details

### Frontend (client/src/app/page.tsx)

```tsx
// State to track failed images
const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());

// Handle image error
const handleImageError = (index: number) => {
  setImageErrors(prev => new Set(prev).add(index));
};

// Render with fallback
{imageErrors.has(index) ? (
  <div className="w-full h-full flex flex-col items-center justify-center p-4">
    <svg className="w-16 h-16 text-gray-400 dark:text-gray-500 mb-3" ...>
      {/* Image icon SVG */}
    </svg>
    <p className="text-xs text-center text-gray-600 dark:text-gray-400 font-mono break-all">
      {result.path}
    </p>
  </div>
) : (
  <img
    src={`${apiUrl}/image/${result.path}`}
    onError={() => handleImageError(index)}
  />
)}
```

### Backend (backend/server.py)

```python
@app.route('/image/<path:image_path>')
def serve_image(image_path):
    # 1. Security check
    if '..' in image_path:
        return jsonify({'error': 'Invalid path'}), 400
    
    # 2. File existence check
    if not os.path.exists(image_path):
        return jsonify({
            'error': 'Image not found',
            'path': image_path,
            'message': 'The image file does not exist'
        }), 404
    
    # 3. File type validation
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png', ...)):
        return jsonify({
            'error': 'Invalid file type',
            'path': image_path
        }), 400
    
    # 4. Try to read and encode
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Auto-detect MIME type
        ext = image_path.lower().split('.')[-1]
        mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
        
        return jsonify({
            'image_data': f"data:{mime_type};base64,{img_base64}",
            'path': image_path
        })
    
    except IOError as e:
        return jsonify({
            'error': 'Cannot read image',
            'path': image_path,
            'message': str(e)
        }), 500
```

## User Experience

### What Users See:

1. **Successful Load:**
   - Normal image display
   - Hover effects work
   - Smooth loading

2. **Failed Load:**
   - Immediate fallback (no broken image icon)
   - Full path visible for debugging
   - Similarity score still shown
   - Card maintains same size and layout

### Benefits:

✅ **No Broken Images:** Users never see browser's broken image icon
✅ **Debugging Info:** File path helps identify missing images
✅ **Consistent Layout:** Failed images don't break the grid
✅ **Graceful Degradation:** App remains functional
✅ **Developer Friendly:** Easy to spot missing files

## Testing

### Test Image Loading:

1. **Create a test with missing image:**
   ```bash
   # Temporarily rename an image
   mv backend/images/123.jpg backend/images/123.jpg.bak
   
   # Search should show fallback
   # Then restore:
   mv backend/images/123.jpg.bak backend/images/123.jpg
   ```

2. **Test with invalid path:**
   ```bash
   curl http://localhost:5000/image/nonexistent.jpg
   # Should return 404 with error details
   ```

3. **Test with invalid file type:**
   ```bash
   curl http://localhost:5000/image/document.pdf
   # Should return 400 with error message
   ```

## Error States Summary

| Scenario | HTTP Status | Frontend Display | User Action |
|----------|-------------|------------------|-------------|
| Image loads successfully | 200 | Image shown | View normally |
| File not found | 404 | Path shown with icon | Check file location |
| Cannot read file | 500 | Path shown with icon | Check permissions |
| Invalid file type | 400 | Path shown with icon | Verify file format |
| Network error | N/A | Path shown with icon | Check backend |

## Accessibility

- **Alt Text:** Set to `Result {index + 1}` for screen readers
- **Error Indication:** Visual icon indicates image unavailable
- **Text Path:** Path is readable by screen readers
- **Color Contrast:** Icon and text meet WCAG standards

## Future Enhancements

Possible improvements:

1. **Retry Mechanism:** Attempt to reload failed images
2. **Placeholder Images:** Show generic placeholder instead of path
3. **Error Details Panel:** Click to see full error message
4. **Batch Validation:** Pre-check image availability before display
5. **Image Caching:** Cache successfully loaded images
6. **Download Link:** Provide download link for valid paths

---

**The application now handles image errors gracefully, ensuring a smooth user experience even when images are missing!** ✨ 