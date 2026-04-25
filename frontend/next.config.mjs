

const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:7860/:path*",
      },
    ];
  },
};

export default nextConfig;
