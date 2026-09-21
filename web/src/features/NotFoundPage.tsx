import { Link } from 'react-router'

export function NotFoundPage() {
  return (
    <div style={{ textAlign: 'center', padding: 'var(--space-9) 0' }}>
      <p className="eyebrow">Dead air</p>
      <h1 style={{ fontSize: 'var(--text-3xl)', margin: 'var(--space-3) 0' }}>
        Nothing on this frequency
      </h1>
      <Link to="/">Back to compose →</Link>
    </div>
  )
}
