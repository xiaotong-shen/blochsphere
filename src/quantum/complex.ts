/** Cartesian complex number. The quantum modules use this instead of JS bigint-free math on pairs. */
export type C = { re: number; im: number };

export function c(re: number, im = 0): C {
  return { re, im };
}

export function add(a: C, b: C): C {
  return { re: a.re + b.re, im: a.im + b.im };
}

export function sub(a: C, b: C): C {
  return { re: a.re - b.re, im: a.im - b.im };
}

export function mul(a: C, b: C): C {
  return {
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re,
  };
}

export function scale(a: C, s: number): C {
  return { re: a.re * s, im: a.im * s };
}

export function conj(a: C): C {
  return { re: a.re, im: -a.im };
}

export function abs2(a: C): number {
  return a.re * a.re + a.im * a.im;
}

export function hypot(a: C): number {
  return Math.hypot(a.re, a.im);
}
