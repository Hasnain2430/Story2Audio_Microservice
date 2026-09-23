/**
 * Routes and providers.
 *
 * Every job has its own URL from the moment it is created. That is not a routing
 * detail — it is the architectural change made visible: v1's results lived inside a
 * Streamlit session and vanished with it, so there was nothing to link to.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router'

import { ApiError } from '@/api/client'
import { Shell } from '@/app/Shell'
import { ComposePage } from '@/features/compose/ComposePage'
import { HistoryPage } from '@/features/history/HistoryPage'
import { JobPage } from '@/features/job/JobPage'
import { NotFoundPage } from '@/features/NotFoundPage'
import { VoicesPage } from '@/features/voices/VoicesPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Retrying a 404 or a validation failure just delays the error the user needs to
      // see. The server already tells us whether a retry could plausibly help.
      retry: (failureCount, error) => {
        if (error instanceof ApiError) return error.retryable && failureCount < 2
        return failureCount < 2
      },
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<ComposePage />} />
            <Route path="jobs/:jobId" element={<JobPage />} />
            <Route path="library" element={<VoicesPage />} />
            <Route path="history" element={<HistoryPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
