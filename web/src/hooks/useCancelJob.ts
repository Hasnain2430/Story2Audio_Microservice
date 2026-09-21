/**
 * Cancelling a running job.
 *
 * Workers check between segments, so this genuinely stops GPU work rather than only
 * relabelling the row -- something v1 could not do at all.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { jobKeys } from '@/hooks/queryKeys'

export function useCancelJob(id: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.cancelJob(id),
    onSuccess: (job) => {
      queryClient.setQueryData(jobKeys.detail(id), job)
      void queryClient.invalidateQueries({ queryKey: jobKeys.all })
    },
  })
}
