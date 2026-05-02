import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Stop } from 'react-native-svg';
import { colors, font } from '@/constants/colors';

const COIN_PALETTES: Record<string, [string, string]> = {
  BTC:   ['#f7931a', '#c5760e'],
  ETH:   ['#627eea', '#3c5fd9'],
  SOL:   ['#14f195', '#9945ff'],
  BNB:   ['#f3ba2f', '#c69323'],
  XRP:   ['#23292f', '#5d6469'],
  ADA:   ['#0033ad', '#1c4dd7'],
  DOGE:  ['#c2a633', '#a08326'],
  AVAX:  ['#e84142', '#b22d2e'],
  MATIC: ['#8247e5', '#5b2ea3'],
  LINK:  ['#2a5ada', '#1d4ab5'],
  DOT:   ['#e6007a', '#a3005a'],
  ATOM:  ['#2e3148', '#5060a8'],
  LTC:   ['#bfbbbb', '#838383'],
  TRX:   ['#ec0928', '#a90719'],
  USDT:  ['#26a17b', '#1c8062'],
  USDC:  ['#2775ca', '#1d5a9c'],
  DEFAULT: ['#22d3ee', '#3b82f6'],
};

function paletteFor(symbol: string): [string, string] {
  const base = symbol.toUpperCase().split(/[\/\-_]/)[0];
  return COIN_PALETTES[base] || COIN_PALETTES.DEFAULT;
}

/**
 * CoinChip — small circular gradient badge with a coin's first letter.
 *
 * Used as a leading visual on strategy cards and trade rows. Avoids loading
 * coin logos (no asset/network round-trip) while still giving each market a
 * distinctive colour identity. Falls back to a cyan→indigo gradient for
 * unknown tickers.
 */
export function CoinChip({
  symbol,
  size = 32,
}: {
  symbol: string;
  size?: number;
}) {
  const uid = React.useId().replace(/:/g, '');
  const gradId = `coin-${uid}`;
  const [c0, c1] = paletteFor(symbol);
  const letter = (symbol.match(/[A-Za-z]/)?.[0] || '?').toUpperCase();

  return (
    <View style={[styles.wrap, { width: size, height: size }]}>
      <Svg width={size} height={size}>
        <Defs>
          <SvgLinearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor={c0} />
            <Stop offset="1" stopColor={c1} />
          </SvgLinearGradient>
        </Defs>
        <Circle cx={size / 2} cy={size / 2} r={size / 2 - 0.5} fill={`url(#${gradId})`} />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={size / 2 - 0.5}
          stroke="rgba(255,255,255,0.18)"
          strokeWidth={1}
          fill="none"
        />
      </Svg>
      <View style={styles.center} pointerEvents="none">
        <Text style={[styles.letter, { fontSize: size * 0.42 }]}>{letter}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { position: 'relative', alignItems: 'center', justifyContent: 'center' },
  center: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  letter: {
    color: colors.text,
    fontFamily: font.black,
    letterSpacing: -0.4,
    textShadowColor: 'rgba(0,0,0,0.35)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
});
