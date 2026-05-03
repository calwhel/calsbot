import React from 'react';

/**
 * Modern-dark theme: no ambient gradient blobs. The screen background is
 * a flat surface (`colors.bg`) so the eye reads content, not decoration.
 * Component is preserved as a no-op so existing imports keep working.
 */
export function AmbientBg(_: { variant?: 'duo' | 'cyan' | 'violet' | 'none' }) {
  return null;
}
