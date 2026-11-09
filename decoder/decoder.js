// Uplink decoder for the 14-byte hive node payload (see hivemon/payload.py for the layout).
// Works as a LoRaWAN network-server codec (decodeUplink) and as a plain Node module.

function s16(hi, lo) {
  var v = (hi << 8) | lo;
  return v & 0x8000 ? v - 0x10000 : v;
}

function decodeUplink(input) {
  var b = input.bytes;
  if (!b || b.length !== 14) {
    return { data: {}, errors: ["payload must be 14 bytes, got " + (b ? b.length : 0)] };
  }
  var version = b[0] >> 4;
  if (version !== 1) {
    return { data: {}, errors: ["unsupported payload version " + version] };
  }
  var round2 = function (x) { return Math.round(x * 100) / 100; };
  return {
    data: {
      alarm: (b[0] & 1) !== 0,
      low_battery: (b[0] & 2) !== 0,
      service: (b[0] & 4) !== 0,
      weight_kg: ((b[1] << 8) | b[2]) / 100,
      t_brood_c: s16(b[3], b[4]) / 100,
      t_amb_c: s16(b[5], b[6]) / 100,
      rh_pct: b[7] / 2,
      tilt_deg: b[8],
      shock_g: round2(b[9] * 0.05),
      battery_v: (2000 + b[10] * 10) / 1000,
      counter: b[11],
      delta30_kg: s16(b[12], b[13]) / 100,
    },
    warnings: [],
  };
}

if (typeof module !== "undefined") {
  module.exports = { decodeUplink: decodeUplink };
}
