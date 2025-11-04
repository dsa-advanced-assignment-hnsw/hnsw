import type { Metadata } from "next";
import "./globals.css";
import { Analytics } from "@vercel/analytics/next"
import { ThemeProvider } from '@/components/theme-provider';

export const metadata: Metadata = {
  title: "HNSW Semantic Search Engine",
  description: "Search images and scientific papers using AI-powered semantic search",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >

          {children}
          <Analytics />
          </ThemeProvider>


      </body>
    </html>
  );
}
