/**
 * Submitting a story.
 *
 * Every submission carries a fresh `Idempotency-Key`, so a retry after a dropped
 * response returns the original job instead of paying for the same generation twice.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import type { CreateJobRequest } from '@/api/types'
import { jobKeys } from '@/hooks/queryKeys'

export function useCreateJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (body: CreateJobRequest) => api.createJob(body, crypto.randomUUID()),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: jobKeys.all })
    },
  })
}
