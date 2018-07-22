/* global sprintf */
/* exported rgb rgba hiddenStyleColor mode FONT_SIZE */

const FONT_SIZE = 13;

/**
 * Returns the rgba string given color and alpha values.
 * @param {[number]} color - in form (r, g, b).
 * @param {number} alpha - the alpha value.
 * @return {string} the rgba string.
 */
function rgba(color, alpha) {
  return sprintf('rgba(%d, %d, %d, %f)', color[0], color[1], color[2], alpha);
}

/**
 * Returns the rgb string given color values.
 * @param {[number]} color - in form (r, g, b).
 * @return {string} the rgb string.
 */
function rgb(color) {
  return sprintf('rgb(%d, %d, %d)', color[0], color[1], color[2]);
}

/**
 * Get the hidden color as rgb, which encodes the id and handle index.
 * @param {int} index - The index.
 * @return {string} - The hidden color rgb string.
 */
function hiddenStyleColor(index) {
  let color = index + 1;
  return rgb([(color >> 16) & 255, (color >> 8) & 255,
    (color & 255)]);
}

/**
 * Function to find mode of a number array.
 * @param {[number]} arr - the array.
 * @return {number} the mode of the array.
 */
function mode(arr) {
  return arr.sort((a, b) =>
      arr.filter((v) => v===a).length
      - arr.filter((v) => v===b).length
  ).pop();
}
