/**
 * A single job.
 *
 * `GET /v1/jobs/{id}` is authoritative at every moment, which is what lets the
 * WebSocket be a pure optimisation. While a job is in flight this also polls slowly as
 * a backstop — if both the socket and the fast poll in `useJobEvents` were to fail, the
 * view still converges rather than freezing on a stale status.
 */

import { useQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { isTerminal } from '@/api/types'
import { jobKeys } from '@/hooks/queryKeys'

export function useJob(id: string | undefined) {
  return useQuery({
    queryKey: jobKeys.detail(id ?? ''),
    queryFn: () => {
      // `enabled` below guarantees this, but asserting it beats a non-null assertion:
      // if the invariant ever breaks, this says so instead of fetching `/v1/jobs/`.
      if (!id) throw new Error('useJob ran without a job id')
      return api.getJob(id)
    },
    enabled: Boolean(id),
    refetchInterval: (query) => {
      const job = query.state.data
      if (!job || isTerminal(job.status)) return false
      return 5000
    },
    // A finished job never changes again, so there is nothing to revalidate.
    staleTime: 2000,
  })
}
