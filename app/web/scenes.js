// Scenes: per-character look of the chat — background, accent colour, fairy-light colours
// and two hand-drawn cliparts for the side panels. Purely decorative; switched off with
// Settings → Appearance → Decorations.

const svg = (body, viewBox = "0 0 160 120") =>
  `<svg viewBox="${viewBox}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${body}</svg>`;

const paperLantern = svg(`
  <line x1="80" y1="2" x2="80" y2="16" stroke="#5a4a3a" stroke-width="2"/>
  <ellipse cx="80" cy="62" rx="44" ry="46" fill="#ff5a3c" opacity=".18"/>
  <rect x="66" y="14" width="28" height="7" rx="2" fill="#3a2a1e"/>
  <ellipse cx="80" cy="60" rx="32" ry="38" fill="#c8281e" stroke="#8c160f" stroke-width="2"/>
  <path d="M64 24 Q52 60 64 96 M80 22 L80 98 M96 24 Q108 60 96 96" fill="none" stroke="#8c160f" stroke-width="1.6"/>
  <ellipse cx="72" cy="48" rx="8" ry="14" fill="#ff8a6a" opacity=".35"/>
  <rect x="66" y="96" width="28" height="7" rx="2" fill="#3a2a1e"/>
  <path d="M80 103 L80 116 M76 116 L84 116" stroke="#f2a93b" stroke-width="2.5" stroke-linecap="round"/>`);

const barrelAndMug = svg(`
  <ellipse cx="74" cy="108" rx="62" ry="7" fill="#000" opacity=".3"/>
  <path d="M18 32 Q10 66 18 104 L86 104 Q94 66 86 32 Z" fill="#7a4a24"/>
  <path d="M36 32 Q32 68 36 104 M52 32 L52 104 M68 32 Q72 68 68 104" stroke="#5e3619" stroke-width="2" fill="none"/>
  <path d="M15 46 Q52 54 89 46 M13 90 Q52 98 91 90" stroke="#3b3b3b" stroke-width="5" fill="none"/>
  <ellipse cx="52" cy="32" rx="34" ry="8" fill="#9a6232"/>
  <ellipse cx="52" cy="32" rx="27" ry="5.5" fill="#5a3418"/>
  <rect x="98" y="58" width="36" height="46" rx="5" fill="#f2a93b" opacity=".9"/>
  <rect x="98" y="58" width="36" height="46" rx="5" fill="none" stroke="#e8e6e1" stroke-width="3" opacity=".55"/>
  <path d="M134 68 h8 q8 0 8 9 v8 q0 9 -8 9 h-8" fill="none" stroke="#e8e6e1" stroke-width="5" opacity=".55"/>
  <circle cx="104" cy="56" r="8" fill="#fff6e6"/><circle cx="116" cy="52" r="10" fill="#fff6e6"/>
  <circle cx="128" cy="56" r="8" fill="#fff6e6"/><circle cx="110" cy="62" r="5" fill="#fff6e6"/>
  <path d="M106 72 v24 M116 70 v26 M126 72 v24" stroke="#ffd27a" stroke-width="2" opacity=".6"/>`);

const candle = svg(`
  <circle cx="80" cy="40" r="34" fill="#ffb347" opacity=".16"/>
  <circle cx="80" cy="40" r="18" fill="#ffcf6a" opacity=".22"/>
  <path d="M80 18 Q90 32 84 42 Q80 48 76 42 Q70 32 80 18 Z" fill="#ffb02e"/>
  <path d="M80 28 Q85 36 82 42 Q80 45 78 42 Q75 36 80 28 Z" fill="#fff1b8"/>
  <line x1="80" y1="44" x2="80" y2="50" stroke="#3a2a1e" stroke-width="2"/>
  <rect x="68" y="50" width="24" height="44" rx="3" fill="#f3e6c8"/>
  <path d="M72 50 v12 q0 4 3 4 q3 0 3 -4 v-12" fill="#fff8e6"/>
  <ellipse cx="80" cy="98" rx="34" ry="7" fill="#8a6a3a"/>
  <ellipse cx="80" cy="95" rx="26" ry="4" fill="#b08a4a"/>
  <path d="M112 96 q14 0 14 -10 q0 -8 -8 -8" fill="none" stroke="#8a6a3a" stroke-width="4"/>`);

const planet = svg(`
  <circle cx="20" cy="18" r="1.5" fill="#fff"/><circle cx="140" cy="12" r="1.2" fill="#cfe3ff"/>
  <circle cx="150" cy="70" r="1.5" fill="#fff"/><circle cx="12" cy="96" r="1.2" fill="#fff"/>
  <path d="M18 76 Q80 98 142 60" fill="none" stroke="#c7b8ff" stroke-width="5" opacity=".55"/>
  <circle cx="78" cy="62" r="34" fill="#4a6fd1"/>
  <path d="M46 52 Q78 44 110 54 M44 66 Q78 60 112 70 M50 80 Q78 76 106 84" stroke="#7fa2f0" stroke-width="5" fill="none" opacity=".7"/>
  <circle cx="66" cy="48" r="9" fill="#9cb8ff" opacity=".35"/>
  <path d="M18 76 Q40 84 60 88" fill="none" stroke="#c7b8ff" stroke-width="5" opacity=".9"/>
  <path d="M104 80 Q126 70 142 60" fill="none" stroke="#c7b8ff" stroke-width="5" opacity=".9"/>
  <circle cx="132" cy="28" r="10" fill="#d9dee8"/><circle cx="128" cy="25" r="2.5" fill="#b4bccb"/>
  <circle cx="135" cy="32" r="2" fill="#b4bccb"/>`);

const rocket = svg(`
  <circle cx="30" cy="20" r="1.4" fill="#fff"/><circle cx="130" cy="30" r="1.6" fill="#fff"/>
  <circle cx="120" cy="100" r="1.2" fill="#cfe3ff"/><circle cx="24" cy="86" r="1.5" fill="#fff"/>
  <g transform="rotate(35 80 60)">
    <path d="M80 8 Q100 30 98 72 L62 72 Q60 30 80 8 Z" fill="#e8ecf4"/>
    <path d="M80 8 Q90 18 94 30 L66 30 Q70 18 80 8 Z" fill="#e5533d"/>
    <circle cx="80" cy="46" r="9" fill="#2b4a8a" stroke="#9fb4d8" stroke-width="3"/>
    <path d="M62 56 L48 80 L62 74 Z M98 56 L112 80 L98 74 Z" fill="#e5533d"/>
    <path d="M68 72 L92 72 L88 80 L72 80 Z" fill="#6d7891"/>
    <path d="M72 80 Q80 112 88 80 Z" fill="#ffb02e"/><path d="M76 80 Q80 98 84 80 Z" fill="#fff1b8"/>
  </g>`);

const books = svg(`
  <ellipse cx="74" cy="110" rx="66" ry="6" fill="#000" opacity=".3"/>
  <rect x="14" y="88" width="110" height="20" rx="3" fill="#6b2d2d"/>
  <rect x="14" y="94" width="110" height="3" fill="#d8c08a" opacity=".7"/>
  <rect x="22" y="68" width="98" height="20" rx="3" fill="#2d4a6b"/>
  <rect x="22" y="74" width="98" height="3" fill="#d8c08a" opacity=".7"/>
  <rect x="18" y="48" width="92" height="20" rx="3" fill="#3f5a33"/>
  <rect x="18" y="54" width="92" height="3" fill="#d8c08a" opacity=".7"/>
  <path d="M110 46 h8 v-8 q0 -6 6 -6 h10 q6 0 6 6 v14 q0 6 -6 6 h-24 z" fill="#2a2a2e"/>
  <rect x="120" y="30" width="16" height="6" rx="2" fill="#4a4a52"/>
  <path d="M128 30 Q120 10 148 2 Q140 18 132 32 Z" fill="#efe6d0"/>
  <path d="M129 30 Q134 16 146 6" stroke="#b9ad90" stroke-width="1.5" fill="none"/>`);

const globe = svg(`
  <circle cx="80" cy="50" r="34" fill="#2f6e8f"/>
  <path d="M62 30 q10 4 14 12 q-6 10 -16 8 q-8 -8 2 -20 Z M88 34 q14 2 18 14 q-4 8 -12 6 q-4 10 -12 4 q2 -12 6 -24 Z M66 66 q10 -2 14 8 q-6 8 -14 4 Z"
        fill="#c9b37a"/>
  <ellipse cx="80" cy="50" rx="34" ry="12" fill="none" stroke="#d8c08a" stroke-width="1" opacity=".5"/>
  <path d="M80 16 Q62 50 80 84 M80 16 Q98 50 80 84" fill="none" stroke="#d8c08a" stroke-width="1" opacity=".5"/>
  <circle cx="68" cy="38" r="8" fill="#ffffff" opacity=".12"/>
  <path d="M40 50 A40 40 0 0 0 120 50" fill="none" stroke="#b08a4a" stroke-width="4"/>
  <line x1="80" y1="90" x2="80" y2="102" stroke="#8a6a3a" stroke-width="5"/>
  <path d="M58 108 Q80 98 102 108 Z" fill="#8a6a3a"/>`);

// Red lantern: a modern bar in the red-light quarter — neon lips, a martini and a stiletto.
const neonLips = svg(`
  <rect x="14" y="10" width="132" height="96" rx="14" fill="#14070b" stroke="#3a1420" stroke-width="3"/>
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="M34 56 C46 34 64 32 80 44 C96 32 114 34 126 56 C110 84 96 92 80 92 C64 92 50 84 34 56 Z"
          stroke="#ff2d6f" stroke-width="12" opacity=".25"/>
    <path d="M34 56 C46 34 64 32 80 44 C96 32 114 34 126 56 C110 84 96 92 80 92 C64 92 50 84 34 56 Z"
          stroke="#ff5c8a" stroke-width="4"/>
    <path d="M40 57 C60 62 100 62 120 57" stroke="#ff9ab8" stroke-width="3"/>
  </g>
  <circle cx="30" cy="24" r="2.5" fill="#ff5c8a"/><circle cx="130" cy="24" r="2.5" fill="#ff5c8a"/>`);

const martini = svg(`
  <ellipse cx="80" cy="56" rx="46" ry="40" fill="#ff2d6f" opacity=".12"/>
  <path d="M38 22 L122 22 L80 66 Z" fill="#ffd9e4" opacity=".25" stroke="#ffe6ee" stroke-width="3" stroke-linejoin="round"/>
  <path d="M50 34 L110 34 L80 62 Z" fill="#ff3d6e" opacity=".75"/>
  <line x1="80" y1="66" x2="80" y2="100" stroke="#ffe6ee" stroke-width="3"/>
  <ellipse cx="80" cy="102" rx="24" ry="5" fill="none" stroke="#ffe6ee" stroke-width="3"/>
  <line x1="98" y1="10" x2="84" y2="42" stroke="#c8a24a" stroke-width="2.5"/>
  <circle cx="84" cy="44" r="7" fill="#d1132f"/><path d="M84 37 C88 30 94 28 98 30" fill="none" stroke="#3d6b3a" stroke-width="2"/>`);

const stiletto = svg(`
  <ellipse cx="80" cy="104" rx="64" ry="7" fill="#000" opacity=".35"/>
  <path d="M26 98 C24 80 34 66 48 62 C62 58 76 60 92 66 C110 72 126 80 134 94 L134 100 L118 100
           C110 88 96 82 84 82 C70 82 60 88 52 100 Z" fill="#c8102e"/>
  <path d="M30 96 C34 76 50 68 66 68" fill="none" stroke="#ff5c7a" stroke-width="3" opacity=".7"/>
  <path d="M120 98 L126 100 L128 112 L124 112 Z" fill="#7a0a1c"/>
  <path d="M26 98 L52 100" stroke="#7a0a1c" stroke-width="4" stroke-linecap="round"/>`);

export const SCENES = {
  plain: { label: "Plain", ring: "#f2a93b", lanterns: true,
           bulbs: ["#f2a93b", "#e5533d", "#ffe3a3", "#f07b2c"], art: [paperLantern, ""] },
  tavern: { label: "Tavern", ring: "#f2b440", lanterns: true,
            bulbs: ["#ffb347", "#ffd27a", "#f07b2c", "#ffe3a3"], art: [barrelAndMug, candle] },
  space: { label: "Space", ring: "#6fb6ff", lanterns: false,
           bulbs: ["#9fd0ff", "#e6f2ff", "#6fb6ff", "#c7b8ff"], art: [planet, rocket] },
  "red-lantern": { label: "Red lantern", ring: "#ff3d6e", lanterns: false,
                   bulbs: ["#ff2d6f", "#ff5c8a", "#c8102e", "#ff9ab8"], art: [neonLips, [martini, stiletto]] },
  study: { label: "Study", ring: "#d8c08a", lanterns: false,
           bulbs: ["#ffe3a3", "#f2c46b", "#fff1cc", "#e0b060"], art: [books, globe] },
};
