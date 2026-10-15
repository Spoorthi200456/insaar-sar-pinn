# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1: deps -- install node_modules in isolation for better layer caching
# ---------------------------------------------------------------------------
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# ---------------------------------------------------------------------------
# Stage 2: builder -- compile the Next.js standalone production server
# ---------------------------------------------------------------------------
FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 3: production -- minimal runtime, non-root user
# ---------------------------------------------------------------------------
FROM node:20-slim AS production
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

RUN groupadd --gid 10001 nextapp \
    && useradd --uid 10001 --gid nextapp --shell /usr/sbin/nologin --no-create-home nextapp

COPY --from=builder --chown=nextapp:nextapp /app/public ./public
COPY --from=builder --chown=nextapp:nextapp /app/.next/standalone ./
COPY --from=builder --chown=nextapp:nextapp /app/.next/static ./.next/static

USER nextapp
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD node -e "fetch('http://localhost:3000').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
