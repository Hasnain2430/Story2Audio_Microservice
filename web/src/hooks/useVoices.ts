/**
 * The voice catalogue: built-ins plus whatever this session has uploaded.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import type { Voice } from '@/api/types'
import { voiceKeys } from '@/hooks/queryKeys'

export function useVoices() {
  return useQuery({
    queryKey: voiceKeys.list(),
    queryFn: () => api.listVoices({ limit: 100 }),
    // Preview URLs are presigned and expire, so this is refetched rather than cached
    // indefinitely -- a stale URL would fail to play with no obvious cause.
    staleTime: 5 * 60_000,
    select: (page): Voice[] => page.items,
  })
}

export function useUploadVoice() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ name, file, filename }: { name: string; file: Blob; filename?: string }) =>
      api.createVoice(name, file, filename),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: voiceKeys.all })
    },
  })
}

export function useDeleteVoice() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => api.deleteVoice(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: voiceKeys.all })
    },
  })
}
