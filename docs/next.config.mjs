import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath: process.env.NODE_ENV === 'production' ? '/consensus' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/consensus/' : '',
  trailingSlash: true,
  
  // GitHub Pages specific
  experimental: {
    // Ensure static export works properly
    appDir: true,
  },
};

export default withMDX(config);