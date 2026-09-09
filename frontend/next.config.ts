import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // `standalone` emits a self-contained server bundle, which keeps the Docker
  // image small; harmless for local development.
  output: 'standalone',
};

export default nextConfig;
