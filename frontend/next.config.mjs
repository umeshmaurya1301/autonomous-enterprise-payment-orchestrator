
const nextConfig = {
  // Export as a fully-static site (no Node.js server required at runtime).
  // `npm run build` produces an `out/` directory that FastAPI serves directly.
  // NOTE: Next.js rewrites/redirects are NOT supported in static export mode,
  // so API calls must target the FastAPI routes directly (no /api/* proxy).
  output: "export",

  // next/image optimisation uses a server-side route that does not exist in
  // static export.  Disable it so <Image> components fall back to plain <img>.
  images: {
    unoptimized: true,
  },

  // Emit a trailing slash so `out/index.html` is served correctly when
  // FastAPI's StaticFiles mounts the directory at "/".
  trailingSlash: true,
};

export default nextConfig;
