import js from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  // `schema.d.ts` is generated from openapi.json by `npm run gen:api`; its style is
  // openapi-typescript's to choose, not ours.
  { ignores: ['dist', 'node_modules', 'src/api/schema.d.ts'] },
  js.configs.recommended,
  {
    // Type-aware rules are scoped to TS sources only. Applying them repo-wide would also
    // pull in this config file itself, which has no tsconfig project and cannot be typed.
    files: ['**/*.{ts,tsx}'],
    extends: [tseslint.configs.strictTypeChecked, tseslint.configs.stylisticTypeChecked],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      '@typescript-eslint/consistent-type-imports': 'error',

      // Two `strictTypeChecked` defaults are tuned rather than obeyed, because their
      // defaults are wrong for React specifically:
      //
      //   `onClick={() => setOpen(true)}` is the idiomatic handler, and the rule's
      //   objection -- that the arrow "returns" the void from the setter -- describes
      //   nothing a reader could misread here.
      '@typescript-eslint/no-confusing-void-expression': [
        'error',
        { ignoreArrowShorthand: true },
      ],
      //   Interpolating a number into a template literal is total and unambiguous;
      //   the rule exists to catch `${someObject}`, which this still catches.
      '@typescript-eslint/restrict-template-expressions': [
        'error',
        { allowNumber: true },
      ],
    },
  },
)
