# F37 Ginger Font

This project uses the **F37 Ginger-Thin** font family, loaded from [OnlineWebFonts.com](https://db.onlinewebfonts.com).

## Font Configuration

The font is automatically loaded via CSS import in `globals.css`:

```css
@import url(https://db.onlinewebfonts.com/c/43d244e16e0aef738eab893861d2f184?family=F37Ginger-Thin);
```

## Font Usage

- **Body Text**: Uses F37 Ginger-Thin
- **Titles & Headings (h1-h6)**: Uses F37 Ginger-Thin with bold weight (700) and wider letter spacing
- **Manual Override**: Use `.font-wide` class to apply bold styling to any element

## Font Weights

Currently using:
- **Regular**: Body text
- **Bold (700)**: Headings and titles

## Styling Details

Headings have additional styling:
- `font-weight: 700` - Bold weight
- `letter-spacing: 0.02em` - Slightly wider letter spacing for better readability

## No Local Files Needed

This font is loaded from a CDN, so no local font files are required in this directory.

The font will load automatically when the application starts.

## Fallback Fonts

If F37 Ginger cannot be loaded, the system will fallback to:
1. `system-ui`
2. `-apple-system`
3. `sans-serif`

## License

The font is loaded from OnlineWebFonts.com. Please verify the license terms for your use case.

## Clean Design

This font was chosen to match the clean, professional design inspired by Aptos Labs:
- No gradients - solid colors only
- Typography-focused design
- Wide letter spacing for modern look
- High readability
