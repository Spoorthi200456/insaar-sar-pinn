/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  transpilePackages: ["@deck.gl/core", "@deck.gl/layers", "@deck.gl/react", "react-map-gl"],
  eslint: {
    ignoreDuringBuilds: false,
  },
};

module.exports = nextConfig;
