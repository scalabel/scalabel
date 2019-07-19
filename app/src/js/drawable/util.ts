import { sprintf } from 'sprintf-js'

// constants
const COLOR_PALETTE = [
  [31, 119, 180],
  [174, 199, 232],
  [255, 127, 14],
  [255, 187, 120],
  [44, 160, 44],
  [152, 223, 138],
  [214, 39, 40],
  [255, 152, 150],
  [148, 103, 189],
  [197, 176, 213],
  [140, 86, 75],
  [196, 156, 148],
  [227, 119, 194],
  [247, 182, 210],
  [127, 127, 127],
  [199, 199, 199],
  [188, 189, 34],
  [219, 219, 141],
  [23, 190, 207],
  [158, 218, 229]
]

/**
 * Tune the shade or tint of rgb color
 * @param {[number,number,number]} rgb: input color
 * @param {[number,number,number]} base: base color (white or black)
 * @param {number} ratio: blending ratio
 * @return {[number,number,number]}
 */
export function blendColor (rgb: number[], base: number[], ratio: number):
 number[] {
  const newRgb = [0, 0, 0]
  for (let i = 0; i < 3; i++) {
    newRgb[i] = Math.max(0,
      Math.min(255, rgb[i] + Math.round((base[i] - rgb[i]) * ratio)))
  }
  return newRgb
}

/**
 * Pick color from the palette. Add additional shades and tints to increase
 * the color number. Results: https://jsfiddle.net/739397/e980vft0/
 * @param {number} index: palette index
 * @return {[number,number,number]}
 */
function pickColorPalette (index: number): number[] {
  const colorIndex = index % COLOR_PALETTE.length
  const shadeIndex = (Math.floor(index / COLOR_PALETTE.length)) % 3
  let rgb = COLOR_PALETTE[colorIndex]
  if (shadeIndex === 1) {
    rgb = blendColor(rgb, [255, 255, 255], 0.4)
  } else if (shadeIndex === 2) {
    rgb = blendColor(rgb, [0, 0, 0], 0.2)
  }
  return rgb
}

/**
 * Convert numerical id to color value
 * @param {number} id
 * @return {number[]}
 */
export function getColorById (id: number): number[] {
  return pickColorPalette(id)
}

/**
 * Convert numerical color to CSS color string
 * @param {number[]} color: can have 3 or 4 elements
 */
export function toCssColor (color: number[]): string {
  if (color.length === 3) {
    return sprintf('rgb(%d, %d, %d)', color[0], color[1], color[2])
  } else if (color.length === 4) {
    return sprintf('rgba(%d, %d, %d, %f)',
                   color[0], color[1], color[2], color[3])
  } else {
    throw new Error(sprintf('color argument has wrong length %d', color.length))
  }
}

export type Context2D = CanvasRenderingContext2D

/* tslint:disable:no-bitwise */

/**
 * Get the label and shape IDs given the control index
 * @param {number} index
 * @return {[number, number]}
 */
export function decodeControlIndex (index: number): number[] {
  return [((index >> 12) & 1023) - 2, index & 1023]
}

/**
 * Get the control color given the label and handle IDs.
 * In the resulting 24bit color, the first 12 bits are for label and the other
 * 12 bits are for handle
 * @param {number} labelId
 * @param {number} handleId
 * @return {number[]}
 */
export function encodeControlColor (
  labelId: number, handleId: number): number[] {
  const index = ((labelId + 2) << 12) | handleId
  return [(index >> 16) & 255, (index >> 8) & 255, (index & 255)]
}

/**
 * Get index from rgb color
 * @param {number[]} color - The rgb color
 * @return {number} - The encoded index.
 */
export function rgbToIndex (color: number[]): number {
  const index = (color[0] << 16) | (color[1] << 8) | (color[2])
  return index
}
