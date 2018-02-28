/* global sprintf */

/* exported Sat SatImage SatLabel */

/*
 Utilities
 */

let COLOR_PALETTE = [
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
  [158, 218, 229],
];

/**
 * Summary: Tune the shade or tint of rgb color
 * @param {[number,number,number]} rgb: input color
 * @param {[number,number,number]} base: base color (white or black)
 * @param {number} ratio: blending ratio
 * @return {[number,number,number]}
 */
function blendColor(rgb, base, ratio) {
  let newRgb = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    newRgb[i] = Math.max(0,
        Math.min(255, rgb[i] + Math.round((base[i] - rgb[i]) * ratio)));
  }
  return newRgb;
}

/**
 * Pick color from the palette. Add additional shades and tints to increase
 * the color number.
 * @param {[int]} index: palette index
 * @return {[number,number,number]}
 */
function pickColorPalette(index) {
  let colorIndex = index % COLOR_PALETTE.length;
  let shadeIndex = (Math.floor(index / COLOR_PALETTE.length)) % 3;
  let rgb = COLOR_PALETTE[colorIndex];
  if (shadeIndex === 1) {
    rgb = blendColor(rgb, [255, 255, 255], 0.4);
  } else if (shadeIndex === 2) {
    rgb = blendColor(rgb, [0, 0, 0], 0.2);
  }
  return rgb;
}

/**
 * Base class for each labeling session/task
 */
function Sat() {
  this.itemList = []; // a.k.a ImageList, but can be 3D model list
  this.currentItem = null;
  this.info = {
    itemUrls: [],
    labels: [],
    numLabeledItems: 0,
    userAgent: null,
  };
}

/**
 * Base class for each labeling target, can be pointcloud or 2D image
 * @param {Sat} sat: context
 */
function SatItem(sat) {
  this.sat = sat;
}


/**
 * Base class for each targeted labeling Image.
 *
 * To define a new tool:
 *
 * function NewTool() {
 *   SatImage.call(this);
 * }
 *
 * NewTool.prototype = Object.create(SatImage.prototype);
 *
 * @param {type} options: tooling options
 */
function SatImage(options) {
  SatItem.call(this, null);
  this.options = options;
  this.index = 0;
}

SatImage.prototype = Object.create(SatItem.prototype);


/**
 * Base class for all the labeled objects.
 *
 * To define a new tool:
 *
 * function NewObject(id) {
 *   SatLabel.call(this, id);
 * }
 *
 * NewObject.prototype = Object.create(SatLabel.prototype);
 *
 * @param {number} id: label object identifier
 * @param {SatItem} satItem: Item that this label appears
 */
function SatLabel(id, satItem = null) {
  this.id = id;
  this.satItem = satItem;
}

/**
 * Pick a color based on the label id
 * @return {(number|number|number)[]}
 */
SatLabel.prototype.color = function() {
  return pickColorPalette(this.id);
};

/**
 * Convert the color to css style
 * @param {number} alpha: color transparency
 * @return {[number,number,number]}
 */
SatLabel.prototype.styleColor = function(alpha = 255) {
  let c = this.color();
  return sprintf('rgba(%d, %d, %d, %f)', c[0], c[1], c[2], alpha);
};
