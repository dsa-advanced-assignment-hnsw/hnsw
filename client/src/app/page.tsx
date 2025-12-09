'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Search, Book, Activity, Github, Linkedin, ArrowUpRight } from 'lucide-react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import Lenis from 'lenis';
import HNSWHeroCanvas from '@/components/hnsw-visuals/HNSWHeroCanvas';
import GraphBackgroundPattern from '@/components/hnsw-visuals/GraphBackgroundPattern';
import FeatureCardWithNodes from '@/components/hnsw-visuals/FeatureCardWithNodes';

export default function LandingPage() {
    const containerRef = useRef(null);
    const [isMobile, setIsMobile] = useState(false);

    // Detect mobile for responsive HNSW canvas configuration
    useEffect(() => {
        const checkMobile = () => {
            setIsMobile(window.innerWidth < 640);
        };
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    // Smooth scroll initialization
    useEffect(() => {
        const lenis = new Lenis({
            duration: 1.2,
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
            orientation: 'vertical',
            gestureOrientation: 'vertical',
            smoothWheel: true,
        });

        function raf(time: number) {
            lenis.raf(time);
            requestAnimationFrame(raf);
        }

        requestAnimationFrame(raf);

        return () => {
            lenis.destroy();
        };
    }, []);

    const marqueeVariants = {
        animate: {
            x: [0, -1000],
            transition: {
                x: {
                    repeat: Infinity,
                    repeatType: "loop" as const,
                    duration: 20,
                    ease: "linear" as const,
                },
            },
        },
    };

    return (
        <div className="min-h-screen bg-[#FFFDF5] text-black font-sans overflow-hidden selection:bg-black selection:text-[#00F0FF]">

            {/* Navigation */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-[#FFFDF5] border-b-4 border-black px-4 py-4">
                <div className="max-w-7xl mx-auto flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <div className="w-10 h-10 bg-blue-600 border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center text-white font-black text-xl">
                            H
                        </div>
                        <span className="text-2xl font-black italic tracking-tighter">HNSW_SEARCH</span>
                    </div>
                    <div className="flex gap-4">
                        <Link
                            href="/search"
                            className="hidden sm:flex px-6 py-2 bg-black text-white font-bold border-2 border-black shadow-[4px_4px_0px_0px_#00F0FF] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_#00F0FF] transition-all items-center gap-2"
                        >
                            LAUNCH APP <ArrowRight className="w-4 h-4" />
                        </Link>
                    </div>
                </div>
            </nav>

            <main className="pt-24">
                {/* Hero Section */}
                <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-4 overflow-hidden">
                    {/* HNSW Graph Visualization Background */}
                    <div className="absolute inset-0 z-0 opacity-30">
                        <HNSWHeroCanvas
                            layerCount={3}
                            baseNodeCount={isMobile ? 30 : 50}
                            animationSpeed={1.0}
                            showTraversal={true}
                            interactive={!isMobile}
                            className="w-full h-full"
                        />
                    </div>
                    
                    <motion.div
                        initial={{ opacity: 0, y: 100 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="text-center relative z-10"
                    >
                        {/* <div className="inline-block px-4 py-2 bg-[#FF00D6] border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] font-mono font-bold mb-8 transform -rotate-2">
                            ⚠️ ALGORITHM V2.0 LIVE
                        </div> */}

                        <h1 className="text-7xl md:text-9xl font-black leading-[0.85] tracking-tighter mb-8">
                            SEARCH<br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-blue-600" style={{ textShadow: '4px 4px 0px #000' }}>
                                BEYOND
                            </span><br />
                            KEYWORDS
                        </h1>

                        <p className="max-w-xl mx-auto text-xl font-bold border-2 border-black p-4 bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] transform rotate-1">
                            Multi-modal search engine powered by HNSW + CLIP.
                            Find images, papers, and fractures with extreme speed.
                        </p>

                        <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mt-12">
                            <Link
                                href="/search"
                                className="w-full sm:w-auto px-8 py-4 bg-blue-600 text-white font-black text-xl border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:bg-blue-500 hover:translate-x-[4px] hover:translate-y-[4px] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all flex items-center justify-center gap-3"
                            >
                                START SEARCHING <ArrowRight className="w-6 h-6" />
                            </Link>
                            <Link
                                href="https://github.com/dsa-advanced-assignment-hnsw/hnsw"
                                target="_blank"
                                className="w-full sm:w-auto px-8 py-4 bg-white text-black font-black text-xl border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:bg-gray-50 hover:translate-x-[4px] hover:translate-y-[4px] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all flex items-center justify-center gap-3"
                            >
                                <Github className="w-6 h-6" /> SOURCE CODE
                            </Link>
                        </div>
                    </motion.div>

                    {/* Decorative Elements */}
                    <div className="absolute top-20 left-10 w-20 h-20 bg-yellow-400 border-4 border-black rounded-full shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] animate-bounce" />
                    <div className="absolute bottom-40 right-10 w-32 h-32 bg-[#00F0FF] border-4 border-black rotate-12 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]" />
                </section>

                {/* Marquee */}
                <div className="bg-black text-[#00F0FF] py-6 border-y-4 border-black overflow-hidden whitespace-nowrap font-black text-4xl">
                    <motion.div animate="animate" variants={marqueeVariants} className="inline-block">
                        IMAGES • RESEARCH PAPERS • MEDICAL DIAGNOSTICS • VECTOR SEARCH • HNSW ALGORITHM • CLIP MODELS • IMAGES • RESEARCH PAPERS • MEDICAL DIAGNOSTICS • VECTOR SEARCH • HNSW ALGORITHM • CLIP MODELS •
                    </motion.div>
                    <motion.div animate="animate" variants={marqueeVariants} className="inline-block">
                        IMAGES • RESEARCH PAPERS • MEDICAL DIAGNOSTICS • VECTOR SEARCH • HNSW ALGORITHM • CLIP MODELS • IMAGES • RESEARCH PAPERS • MEDICAL DIAGNOSTICS • VECTOR SEARCH • HNSW ALGORITHM • CLIP MODELS •
                    </motion.div>
                </div>

                {/* Features Section */}
                <section className="py-24 px-4 bg-blue-600 border-b-4 border-black relative overflow-hidden">
                    {/* Graph Background Pattern with Parallax */}
                    <GraphBackgroundPattern
                        density="sparse"
                        animated={true}
                        parallaxIntensity={0.3}
                        className="opacity-20"
                    />
                    
                    <div className="max-w-7xl mx-auto relative z-10">
                        <h2 className="text-6xl font-black text-white mb-16 text-center transform -rotate-1" style={{ textShadow: '6px 6px 0px #000' }}>
                            POWERED BY AI
                        </h2>

                        <div className="grid md:grid-cols-3 gap-8">
                            {/* Card 1 - Image Search */}
                            <FeatureCardWithNodes
                                icon={<Search className="w-8 h-8" />}
                                title="Image Search"
                                description="Natural language queries via CLIP ViT-B/32. Find 'sunset' without tags."
                                accentColor="#FF00D6"
                            >
                                <div className="mt-4">
                                    <div className="h-3 bg-black w-full mb-2 rounded"></div>
                                    <div className="h-3 bg-gray-300 w-3/4 rounded"></div>
                                </div>
                            </FeatureCardWithNodes>

                            {/* Card 2 - Paper Search */}
                            <FeatureCardWithNodes
                                icon={<Book className="w-8 h-8" />}
                                title="Paper Search"
                                description="1M+ arXiv papers. Semantic matching with RoBERTa-large."
                                accentColor="#00F0FF"
                            >
                                <div className="mt-4">
                                    <div className="h-3 bg-black w-full mb-2 rounded"></div>
                                    <div className="h-3 bg-gray-300 w-3/4 rounded"></div>
                                </div>
                            </FeatureCardWithNodes>

                            {/* Card 3 - Medical Scan */}
                            <FeatureCardWithNodes
                                icon={<Activity className="w-8 h-8" />}
                                title="Medical Scan"
                                description="Bone fracture detection using BiomedCLIP. Privacy-first analysis."
                                accentColor="#FACC15"
                            >
                                <div className="mt-4">
                                    <div className="h-3 bg-black w-full mb-2 rounded"></div>
                                    <div className="h-3 bg-gray-300 w-3/4 rounded"></div>
                                </div>
                            </FeatureCardWithNodes>
                        </div>
                    </div>
                </section>

                {/* Team Section - Equal Cards with Avatars */}
                <section className="py-24 px-4 bg-[#FFFDF5]">
                    <div className="max-w-6xl mx-auto">
                        <h2 className="text-6xl font-black text-black mb-16 text-center uppercase tracking-tight">
                            The Builders
                        </h2>

                        {/* 3 Equal Cards Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {/* Card 1 - Huy Pham */}
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5 }}
                                className="bg-[#FF00D6] rounded-3xl p-6 border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex flex-col"
                            >
                                {/* Avatar */}
                                <div className="w-24 h-24 rounded-full border-4 border-black mb-4 overflow-hidden relative">
                                    <Image
                                        src="/anderson-avt.jpg"
                                        alt="Huy Pham"
                                        fill
                                        className="object-cover"
                                    />
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-2xl font-black text-black mb-1">Huy Pham</h3>
                                    <p className="text-black/80 font-bold text-sm mb-4">Full stack developer building the complete search experience.</p>
                                </div>
                                <div className="flex items-end justify-between mt-auto pt-4">
                                    <span className="text-black/60 font-mono text-xs">FULL STACK</span>
                                    <div className="flex gap-2">
                                        <Link href="https://github.com/huyphamcs" target="_blank" className="w-9 h-9 bg-black rounded-full flex items-center justify-center hover:scale-110 transition-transform">
                                            <Github className="w-4 h-4 text-white" />
                                        </Link>
                                        <Link href="#" className="w-9 h-9 bg-black rounded-full flex items-center justify-center hover:scale-110 transition-transform">
                                            <Linkedin className="w-4 h-4 text-white" />
                                        </Link>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Card 2 - Nguyen Dinh Huy */}
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5, delay: 0.1 }}
                                className="bg-[#00F0FF] rounded-3xl p-6 border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex flex-col"
                            >
                                {/* Avatar */}
                                <div className="w-24 h-24 bg-black/20 rounded-full border-4 border-black mb-4 flex items-center justify-center overflow-hidden">
                                    <span className="text-4xl font-black text-black/40">NH</span>
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-2xl font-black text-black mb-1">Nguyen Dinh Huy</h3>
                                    <p className="text-black/80 font-bold text-sm mb-4">Backend systems & DevOps infrastructure.</p>
                                </div>
                                <div className="flex items-end justify-between mt-auto pt-4">
                                    <span className="text-black/60 font-mono text-xs">BACKEND & DEVOPS</span>
                                    <div className="flex gap-2">
                                        <Link href="https://github.com/huynguyen6906" target="_blank" className="w-9 h-9 bg-black rounded-full flex items-center justify-center hover:scale-110 transition-transform">
                                            <Github className="w-4 h-4 text-white" />
                                        </Link>
                                        <Link href="#" className="w-9 h-9 bg-black rounded-full flex items-center justify-center hover:scale-110 transition-transform">
                                            <Linkedin className="w-4 h-4 text-white" />
                                        </Link>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Card 3 - Tran Quang Huy */}
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5, delay: 0.2 }}
                                className="bg-[#FACC15] rounded-3xl p-6 border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex flex-col"
                            >
                                {/* Avatar */}
                                <div className="w-24 h-24 bg-black/20 rounded-full border-4 border-black mb-4 flex items-center justify-center overflow-hidden">
                                    <span className="text-4xl font-black text-black/40">TH</span>
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-2xl font-black text-black mb-1">Tran Quang Huy</h3>
                                    <p className="text-black/80 font-bold text-sm mb-4">Platform engineering & system architecture.</p>
                                </div>
                                <div className="flex items-end justify-between mt-auto pt-4">
                                    <span className="text-black/60 font-mono text-xs">PLATFORM ENGINEER</span>
                                    <div className="flex gap-2">
                                        <Link href="https://github.com/BofMeAstaroth" target="_blank" className="w-9 h-9 bg-black rounded-full flex items-center justify-center hover:scale-110 transition-transform">
                                            <Github className="w-4 h-4 text-white" />
                                        </Link>
                                        <Link href="#" className="w-9 h-9 bg-black rounded-full flex items-center justify-center hover:scale-110 transition-transform">
                                            <Linkedin className="w-4 h-4 text-white" />
                                        </Link>
                                    </div>
                                </div>
                            </motion.div>
                        </div>
                    </div>
                </section>

            </main>

            <footer className="bg-black text-white py-12 border-t-4 border-white">
                <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center">
                    <div className="flex items-center gap-4 mb-8 md:mb-0">
                        <div className="w-12 h-12 bg-white border-4 border-blue-600 flex items-center justify-center text-black font-black text-2xl">H</div>
                        <span className="text-2xl font-black">HNSW_SEARCH</span>
                    </div>
                    <div className="text-right font-mono text-gray-400">
                        © 2025 ALL RIGHTS RESERVED<br />
                        BUILT WITH RAGE AND CODE
                    </div>
                </div>
            </footer>
        </div>
    );
}
