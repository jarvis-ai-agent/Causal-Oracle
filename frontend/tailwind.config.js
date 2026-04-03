/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: '#0a0e17',
          surface: '#0f1622',
          panel: '#141d2e',
          border: '#1e2d45',
          accent: '#00d4ff',
          gold: '#fbbf24',
          green: '#22c55e',
          red: '#ef4444',
          orange: '#f97316',
          text: '#c9d1e0',
          muted: '#5a6a7e',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
