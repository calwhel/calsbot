/**
 * TradeHub mobile design tokens — Modern Dark.
 *
 * Aesthetic: calm graphite surfaces, single restrained green accent for
 * positive/active states, single restrained red for negative. No glows,
 * no gradient meshes, no per-coin colour identities. Hairline borders.
 *
 * Two-layer system:
 *   - `colors`   — semantic surfaces, text, and tonal palettes
 *   - `gradient` — kept for API compatibility; values are flat or near-flat
 *   - `glow`     — kept for API compatibility; values render as flat (no shadow)
 *   - `radius`   — corner radii (tightened)
 *   - `spacing`  — layout rhythm
 *   - `font`     — Inter family PostScript names
 *
 * IMPORTANT: All historical keys (violet, gold, magenta, mint, indigo, etc.)
 * are preserved as aliases that resolve to neutral text/dim values. Screens
 * that still pass tone="violet" will simply render in the calm neutral tone
 * — no breakage, no rainbow.
 */

// Core palette — neutral graphite, no blue undertone.
// Bg is pulled darker so cards lift visibly off the surface.
const SURFACE_0  = '#08090B';   // base background (near-black)
const SURFACE_1  = '#101115';   // elevated section background
const SURFACE_2  = '#1A1C20';   // standard card surface
const SURFACE_3  = '#23262C';   // hover / pressed / chip surface

const HAIRLINE   = 'rgba(255,255,255,0.06)';
const HAIRLINE_2 = 'rgba(255,255,255,0.10)';

const TEXT_HI    = '#F2F2F3';
const TEXT_MID   = '#9A9BA0';
const TEXT_LOW   = '#65676E';

// Single restrained green accent (Coinbase-ish, not neon)
const GREEN      = '#3FB68B';
const GREEN_SOFT = 'rgba(63,182,139,0.14)';

// Single restrained red for negative
const RED        = '#E5484D';
const RED_SOFT   = 'rgba(229,72,77,0.14)';

// Muted amber for warnings (not yellow, not gold)
const AMBER      = '#D6A35C';
const AMBER_SOFT = 'rgba(214,163,92,0.14)';

export const colors = {
  // Surfaces
  bg:        SURFACE_0,
  bgElev:    SURFACE_1,
  card:      SURFACE_2,
  cardHi:    SURFACE_3,
  border:    HAIRLINE,
  borderHi:  HAIRLINE_2,
  borderAccent: HAIRLINE_2,

  // Text
  text:      TEXT_HI,
  textDim:   TEXT_MID,
  textMute:  TEXT_LOW,

  // Brand accent — single restrained green
  accent:    GREEN,
  accentSoft: GREEN,
  accentDim: GREEN_SOFT,
  accentText: '#0E0F11',

  // Tonals
  positive:    GREEN,
  positiveDim: GREEN_SOFT,
  negative:    RED,
  negativeDim: RED_SOFT,
  warning:     AMBER,
  warningDim:  AMBER_SOFT,

  // Legacy extended palette — aliased to neutral text so historical
  // tone props (violet/gold/magenta/mint/indigo) render quietly.
  violet:    TEXT_HI,
  violetDim: SURFACE_3,
  gold:        AMBER,
  goldDim:     AMBER_SOFT,
  magenta:     TEXT_HI,
  magentaDim:  SURFACE_3,
  mint:        GREEN,
  mintDim:     GREEN_SOFT,
  indigo:      TEXT_HI,
  indigoDim:   SURFACE_3,

  // Misc
  pillBg:    SURFACE_3,
  glassBg:   SURFACE_2,
  glassHi:   SURFACE_3,
  divider:   HAIRLINE,
} as const;

/**
 * Gradient stop pairs — preserved for API compatibility. Each "gradient"
 * is now two near-identical stops so any consumer SVG renders as a flat
 * surface in the modern-dark tone.
 */
export const gradient = {
  brand:     [GREEN, GREEN, GREEN] as const,
  brandSoft: [SURFACE_2, SURFACE_2, SURFACE_2] as const,
  card:      [SURFACE_2, SURFACE_2, SURFACE_2] as const,
  cardHi:    [SURFACE_3, SURFACE_3, SURFACE_3] as const,
  positive:  [GREEN, GREEN] as const,
  negative:  [RED, RED] as const,
  warning:   [AMBER, AMBER] as const,
  violet:    [TEXT_HI, TEXT_HI] as const,
  ambCyan:   ['rgba(0,0,0,0)', 'rgba(0,0,0,0)'] as const,
  ambViolet: ['rgba(0,0,0,0)', 'rgba(0,0,0,0)'] as const,
} as const;

/**
 * Elevation shadows — these are TRUE shadows (pure black, blurred, low
 * opacity) for premium card lift. NOT coloured glows. They give depth
 * without tinting. The tonal `accent` / `positive` presets render no
 * glow (still flat) — coloured glow is what cheapens dark UIs.
 */
export const glow = {
  accent: {
    shadowColor: 'transparent',
    shadowOpacity: 0,
    shadowRadius: 0,
    shadowOffset: { width: 0, height: 0 },
    elevation: 0,
  },
  positive: {
    shadowColor: 'transparent',
    shadowOpacity: 0,
    shadowRadius: 0,
    shadowOffset: { width: 0, height: 0 },
    elevation: 0,
  },
  card: {
    shadowColor: '#000000',
    shadowOpacity: 0.35,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 6 },
    elevation: 4,
  },
  pill: {
    shadowColor: 'transparent',
    shadowOpacity: 0,
    shadowRadius: 0,
    shadowOffset: { width: 0, height: 0 },
    elevation: 0,
  },
} as const;

/** Discrete elevation steps — true black shadows, no tint. */
export const shadow = {
  none: {
    shadowColor: 'transparent', shadowOpacity: 0, shadowRadius: 0,
    shadowOffset: { width: 0, height: 0 }, elevation: 0,
  },
  card: {
    shadowColor: '#000000', shadowOpacity: 0.35, shadowRadius: 18,
    shadowOffset: { width: 0, height: 6 }, elevation: 4,
  },
  lift: {
    shadowColor: '#000000', shadowOpacity: 0.45, shadowRadius: 28,
    shadowOffset: { width: 0, height: 12 }, elevation: 8,
  },
} as const;

export const radius = {
  sm: 6,
  md: 10,
  lg: 12,
  xl: 16,
  pill: 999,
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
} as const;

/** Inter PostScript names. Match what's loaded in app/_layout.tsx via useFonts. */
export const font = {
  regular:  'Inter_400Regular',
  medium:   'Inter_500Medium',
  semibold: 'Inter_600SemiBold',
  bold:     'Inter_700Bold',
  black:    'Inter_700Bold', // alias — modern-dark never uses ExtraBold
} as const;
