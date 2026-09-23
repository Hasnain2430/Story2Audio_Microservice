/**
 * Interface primitives.
 *
 * Small, unopinionated, and styled entirely through `primitives.css` tokens. Every one
 * of these exists because it appears in at least two places — nothing here is a wrapper
 * for its own sake.
 */

import type { ButtonHTMLAttributes, CSSProperties, ReactNode } from 'react'

import type { JobStatus } from '@/api/types'
import { STATUS_LABEL } from '@/lib/format'
import '@/components/primitives.css'

// --- Button ---------------------------------------------------------------------------

type ButtonVariant = 'primary' | 'ghost' | 'quiet' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: 'sm' | 'md' | 'lg'
  busy?: boolean
}

export function Button({
  variant = 'ghost',
  size = 'md',
  busy = false,
  className = '',
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      className={`btn btn--${variant} btn--${size} ${className}`.trim()}
      // A busy button must not be clickable, and must say so to a screen reader rather
      // than only looking different.
      disabled={disabled === true || busy}
      aria-busy={busy || undefined}
      {...rest}
    >
      {busy && <span className="btn__spinner" aria-hidden="true" />}
      {children}
    </button>
  )
}

// --- Field ----------------------------------------------------------------------------

interface FieldProps {
  label: string
  hint?: string
  htmlFor?: string
  error?: string
  children: ReactNode
}

export function Field({ label, hint, htmlFor, error, children }: FieldProps) {
  return (
    <div className="field">
      <label className="field__label eyebrow" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {error ? (
        <p className="field__error" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p className="field__hint">{hint}</p>
      ) : null}
    </div>
  )
}

// --- Segmented control -------------------------------------------------------------------

interface SegmentedOption<T extends string> {
  value: T
  label: string
  hint?: string
}

interface SegmentedProps<T extends string> {
  name: string
  value: T
  options: readonly SegmentedOption<T>[]
  onChange: (value: T) => void
}

/**
 * A radio group that looks like a control surface.
 *
 * Built from real radio inputs rather than buttons, so arrow keys move between options
 * and the whole group is one tab stop — the behaviour a keyboard user expects.
 */
export function Segmented<T extends string>({ name, value, options, onChange }: SegmentedProps<T>) {
  return (
    <div className="segmented" role="radiogroup" aria-label={name}>
      {options.map((option) => (
        <label
          key={option.value}
          className={`segmented__option ${value === option.value ? 'is-selected' : ''}`}
        >
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
            className="visually-hidden"
          />
          <span className="segmented__label">{option.label}</span>
          {option.hint && <span className="segmented__hint">{option.hint}</span>}
        </label>
      ))}
    </div>
  )
}

// --- Status light --------------------------------------------------------------------------

/**
 * Job status as a lamp rather than a coloured pill.
 *
 * An in-flight job pulses; a finished one is steady. Colour is never the only signal —
 * the label is always present — so this still reads without colour vision.
 */
export function StatusLight({ status, label = true }: { status: JobStatus; label?: boolean }) {
  const active = status === 'writing' || status === 'synthesizing' || status === 'queued'

  return (
    <span className={`status status--${status}`}>
      <span className="status__lamp" aria-hidden="true">
        {active && <span className="status__ring" />}
      </span>
      {label && <span className="status__text">{STATUS_LABEL[status]}</span>}
    </span>
  )
}

// --- Segment meter ----------------------------------------------------------------------------

/**
 * Synthesis progress as discrete ticks, one per segment.
 *
 * Deliberately not a continuous bar: the work really is quantised into segments, and
 * showing the actual count is more honest — and more reassuring on a long job — than a
 * smooth fill that implies knowledge we do not have. v1 could show nothing at all here,
 * because it synthesised the whole story in a single call.
 */
export function SegmentMeter({ done, total }: { done: number; total: number }) {
  if (total <= 0) return null

  // Above this, individual ticks stop being legible and a bar reads better.
  const ticks = total <= 48 ? total : 0

  return (
    <div
      className="meter"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={done}
      aria-label={`${done} of ${total} segments recorded`}
    >
      {ticks > 0 ? (
        <div className="meter__ticks">
          {Array.from({ length: ticks }, (_, index) => (
            <span
              key={index}
              className={`meter__tick ${index < done ? 'is-on' : ''} ${
                index === done ? 'is-next' : ''
              }`}
            />
          ))}
        </div>
      ) : (
        <div className="meter__bar">
          <div className="meter__fill" style={{ width: `${(done / total) * 100}%` }} />
        </div>
      )}
      <span className="meter__count mono">
        {done}/{total}
      </span>
    </div>
  )
}

// --- Misc --------------------------------------------------------------------------------------

export function Card({
  children,
  className = '',
  style,
  as: Tag = 'div',
}: {
  children: ReactNode
  className?: string
  style?: CSSProperties
  as?: 'div' | 'section' | 'article' | 'li'
}) {
  return (
    <Tag className={`card ${className}`.trim()} style={style}>
      {children}
    </Tag>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <p className="empty__title">{title}</p>
      {children && <div className="empty__body">{children}</div>}
    </div>
  )
}

export function Banner({
  tone = 'info',
  children,
}: {
  tone?: 'info' | 'error' | 'warn'
  children: ReactNode
}) {
  return (
    <div className={`banner banner--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      {children}
    </div>
  )
}
