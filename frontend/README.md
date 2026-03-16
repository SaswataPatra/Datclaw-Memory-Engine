# DAPPY Frontend

Modern chat interface for DAPPY cognitive memory system.

## Tech Stack

- **React 18** with TypeScript
- **Vite** for blazing fast dev experience
- **Tailwind CSS v4** for styling
- **React Router** for navigation

## Getting Started

```bash
# Install dependencies
npm install

# Start dev server (proxies API to localhost:8001)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Features

- JWT-based authentication
- Real-time chat interface
- Responsive design
- Auto-resizing textarea
- Loading states and error handling

## API Integration

The dev server proxies `/api/*` requests to `http://localhost:8001` (the backend).

In production, configure `VITE_API_BASE_URL` environment variable if needed.
