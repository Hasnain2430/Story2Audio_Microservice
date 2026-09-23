/**
 * Theme preference.
 *
 * Three states, matching the platform convention: an explicit choice stamps
 * `data-theme` on the root element, and "system" stamps nothing so
 * `prefers-color-scheme` decides. Persisted per browser, which is a convenience and
 * nothing more -- every read and write is guarded, because a private window or a
 * browser set to block site data throws on access rather than returning null.
 */

import { useCallback, useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 's2a.theme'

function systemTheme(): Theme {
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function readStored(): Theme | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY)
    return value === 'dark' || value === 'light' ? value : null
  } catch {
    return null
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => readStored() ?? systemTheme())

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      window.localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      // Storage is unavailable. The theme still applies for this session.
    }
  }, [theme])

  const toggle = useCallback(() => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, toggle }
}
