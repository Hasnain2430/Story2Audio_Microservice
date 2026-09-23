/**
 * Query keys.
 *
 * Centralised so invalidation cannot silently miss a cache entry: a hook and the code
 * that invalidates it must agree on the exact key, and that agreement is easy to break
 * when the arrays are written inline at both ends.
 */

export const jobKeys = {
  all: ['jobs'] as const,
  list: (cursor?: string) => ['jobs', 'list', cursor ?? null] as const,
  detail: (id: string) => ['jobs', 'detail', id] as const,
}

export const voiceKeys = {
  all: ['voices'] as const,
  list: () => ['voices', 'list'] as const,
}
