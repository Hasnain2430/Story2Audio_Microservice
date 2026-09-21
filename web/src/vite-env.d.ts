/// <reference types="vite/client" />

/**
 * Typed environment.
 *
 * Without this, `import.meta.env.VITE_*` is `any`, which silently defeats the strict
 * type checking everywhere it is used.
 */
interface ImportMetaEnv {
  readonly VITE_API_ORIGIN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
