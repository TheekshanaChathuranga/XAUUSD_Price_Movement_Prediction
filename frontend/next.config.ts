import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        // Proxy to your Python FastAPI backend
        destination: process.env.API_URL 
          ? `${process.env.API_URL}/api/:path*` 
          : "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
