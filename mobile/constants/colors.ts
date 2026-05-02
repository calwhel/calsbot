/**
 * TradeHub mobile design tokens.
 *
 * Two-layer system:
 *   - `colors`   — semantic surfaces, text, and tonal palettes
 *   - `gradient` — paired stops used by SVG/LinearGradient backdrops
 *   - `glow`     — soft shadow presets for elevated/accented elements
 *   - `radius`   — corner radii
 *   - `spacing`  — layout rhythm
 *   - `font`     — Inter family PostScript names (paired with @expo-google-fonts/inter)
 */

export const colors = {
  // Surfaces — slightly deeper than v1 with a hint of indigo undertone
  bg:        '#070912',
  bgElev:    '#0f1424',
  card:      '#141a2c',
  cardHi:    '#1a2238',
  border:    '#212a44',
  borderHi:  '#2c365a',
  borderAccent: 'rgba(34, 211, 238, 0.28)',

  // Text
  text:      '#f4f6fb',
  textDim:   '#a3acc7',
  textMute:  '#6b7392',

  // Brand accent (cyan)
  accent:    '#22d3ee',
  accentSoft:'#67e8f9',
  accentDim: 'rgba(34, 211, 238, 0.16)',
  accentText:'#04111a',

  // Secondary accent (violet) — used for highlights, sparingly
  violet:    '#a78bfa',
  violetDim: 'rgba(167, 139, 250, 0.14)',

  // Tonals
  positive:    '#34d399',
  positiveDim: 'rgba(52, 211, 153, 0.16)',
  negative:    '#f87171',
  negativeDim: 'rgba(248, 113, 113, 0.16)',
  warning:     '#fbbf24',
  warningDim:  'rgba(251, 191, 36, 0.16)',

  // Extended palette — used for bento tile variety + category chips
  gold:        '#f5b754',
  goldDim:     'rgba(245, 183, 84, 0.14)',
  magenta:     '#f472b6',
  magentaDim:  'rgba(244, 114, 182, 0.14)',
  mint:        '#5eead4',
  mintDim:     'rgba(94, 234, 212, 0.14)',
  indigo:      '#6366f1',
  indigoDim:   'rgba(99, 102, 241, 0.16)',

  // Misc
  pillBg:    '#1c2338',
  glassBg:   'rgba(20, 26, 44, 0.72)',
  glassHi:   'rgba(28, 36, 60, 0.78)',
  divider:   'rgba(255, 255, 255, 0.06)',
} as const;

/** Gradient stop pairs — consumed by `<LinearGradient>` in svg primitives. */
export const gradient = {
  // Hero cyan→indigo (used on hero P&L card + primary CTAs)
  brand:    ['#22d3ee', '#3b82f6', '#6366f1'] as const,
  brandSoft:['#1d3a4f', '#1a2452', '#221a4a'] as const,

  // Card gradient — vertical fade from elevated to base
  card:     ['#1a2238', '#141b2e', '#0f1524'] as const,
  cardHi:   ['#1f2945', '#172036', '#111729'] as const,

  // Tone-coded
  positive: ['#34d399', '#10b981'] as const,
  negative: ['#f87171', '#ef4444'] as const,
  warning:  ['#fbbf24', '#f59e0b'] as const,
  violet:   ['#a78bfa', '#7c3aed'] as const,

  // Ambient screen backdrop blobs
  ambCyan:  ['rgba(34,211,238,0.20)', 'rgba(34,211,238,0)'] as const,
  ambViolet:['rgba(139,92,246,0.16)', 'rgba(139,92,246,0)'] as const,
} as const;

/** Soft shadow presets. iOS only — Android uses elevation as a fallback. */
export const glow = {
  accent: {
    shadowColor: '#22d3ee',
    shadowOpacity: 0.35,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 8 },
    elevation: 10,
  },
  positive: {
    shadowColor: '#34d399',
    shadowOpacity: 0.30,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 6 },
    elevation: 8,
  },
  card: {
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
    elevation: 6,
  },
  pill: {
    shadowColor: '#000',
    shadowOpacity: 0.20,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 22,
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
  black:    'Inter_800ExtraBold',
} as const;
