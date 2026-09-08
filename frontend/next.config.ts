import type { NextConfig } from 'next';

/** Headers for pages and routes that handle identity; never cached, never framed. */
const IDENTITY_HEADERS = [
  { key: 'Cache-Control', value: 'no-store' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'same-origin' },
];

const nextConfig: NextConfig = {
  // Enable standalone output for Docker deployment
  output: 'standalone',
  async headers() {
    return [
      { source: '/admin', headers: IDENTITY_HEADERS },
      { source: '/admin/:path*', headers: IDENTITY_HEADERS },
      { source: '/account', headers: IDENTITY_HEADERS },
      { source: '/account/:path*', headers: IDENTITY_HEADERS },
      { source: '/login', headers: IDENTITY_HEADERS },
      { source: '/login/:path*', headers: IDENTITY_HEADERS },
      { source: '/change-password', headers: IDENTITY_HEADERS },
      { source: '/change-password/:path*', headers: IDENTITY_HEADERS },
      { source: '/api/:path*', headers: IDENTITY_HEADERS },
    ];
  },
};

export default nextConfig;
