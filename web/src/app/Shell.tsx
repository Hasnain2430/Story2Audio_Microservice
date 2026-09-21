/**
 * Application shell: masthead, navigation, theme control.
 *
 * The masthead is styled like a station identification rather than a product logo —
 * this is a thing that reads to you, so it should feel like something you tune into.
 */

import { NavLink, Outlet } from 'react-router'

import { useTheme } from '@/app/useTheme'
import '@/app/Shell.css'

const NAV = [
  { to: '/', label: 'Compose', end: true },
  { to: '/library', label: 'Voices', end: false },
  { to: '/history', label: 'History', end: false },
]

export function Shell() {
  const { theme, toggle } = useTheme()

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead__inner">
          <NavLink to="/" className="brand" aria-label="Story2Audio, home">
            <span className="brand__mark" aria-hidden="true">
              <span className="brand__dot" />
            </span>
            <span className="brand__text">
              <span className="brand__name">Story2Audio</span>
              <span className="brand__tag eyebrow">Narration on demand</span>
            </span>
          </NavLink>

          <nav className="nav" aria-label="Primary">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => `nav__link ${isActive ? 'is-active' : ''}`}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <button
            type="button"
            className="theme-toggle"
            onClick={toggle}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </header>

      <main className="shell__main">
        <Outlet />
      </main>

      <footer className="shell__footer">
        <p className="mono">
          Queue-backed pipeline · story and speech generated asynchronously · v2
        </p>
      </footer>
    </div>
  )
}

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.4" />
      <g stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
        <path d="M8 1v1.8M8 13.2V15M15 8h-1.8M2.8 8H1M12.9 3.1l-1.3 1.3M4.4 11.6l-1.3 1.3M12.9 12.9l-1.3-1.3M4.4 4.4 3.1 3.1" />
      </g>
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path
        d="M13.5 9.8A6 6 0 0 1 6.2 2.5a6 6 0 1 0 7.3 7.3Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  )
}
