# ── Stage 1: build Vite/React app ────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (layer caching: only re-run if package files change)
# NOTE: using npm install instead of npm ci so package-lock.json is not required.
# Once the frontend developer runs `npm install` locally and commits package-lock.json,
# switch this back to:
#   COPY package.json package-lock.json ./
#   RUN npm ci --frozen-lockfile
COPY package.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build
# Output: /app/dist

# ── Stage 2: serve via nginx ──────────────────────────────────────────────────
FROM nginx:1.25-alpine AS runtime

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# SPA routing: 404s fall back to index.html so React Router works
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
