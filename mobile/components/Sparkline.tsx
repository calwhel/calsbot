import React from 'react';
import Svg, { Defs, LinearGradient, Stop, Path } from 'react-native-svg';
import { colors } from '@/constants/colors';

/**
 * Tiny inline sparkline for list rows. Auto-scales to its data, fills under
 * the curve, picks colour from final-value sign.
 *
 * If `values` has < 2 points, returns null (caller should render a placeholder).
 */
export function Sparkline({
  values,
  width = 72,
  height = 28,
  color,
  strokeWidth = 1.6,
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  strokeWidth?: number;
}) {
  // Hooks must run unconditionally — declare the per-instance ID BEFORE the
  // early-return guard so React's hook order stays stable between renders.
  const uid = React.useId().replace(/:/g, '');
  const gradId = `spark-${uid}`;

  if (!values || values.length < 2) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const dx = width / (values.length - 1);

  const pts = values.map((v, i) => {
    const x = i * dx;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return { x, y };
  });

  const linePath = pts.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');
  const fillPath = `${linePath} L ${pts[pts.length - 1].x} ${height} L ${pts[0].x} ${height} Z`;

  const last = values[values.length - 1];
  const first = values[0];
  const stroke =
    color ??
    (last > first ? colors.positive : last < first ? colors.negative : colors.textDim);

  return (
    <Svg width={width} height={height}>
      <Defs>
        <LinearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0" stopColor={stroke} stopOpacity={0.32} />
          <Stop offset="1" stopColor={stroke} stopOpacity={0} />
        </LinearGradient>
      </Defs>
      <Path d={fillPath} fill={`url(#${gradId})`} />
      <Path
        d={linePath}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </Svg>
  );
}
