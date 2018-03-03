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
 * the color number. Results: https://jsfiddle.net/739397/e980vft0/
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
 * @param {SatItem} ItemType: item instantiation type
 * @param {SatLabel} LabelType: label instantiation type
 */
function Sat(ItemType, LabelType) {
  this.items = []; // a.k.a ImageList, but can be 3D model list
  this.labels = []; // list of label objects
  this.labelIdMap = {};
  this.lastLabelId = 0;
  this.currentItem = null;
  this.ItemType = ItemType;
  this.LabelType = LabelType;
  this.events = [];
  this.startTime = Date().now();
}

Sat.prototype.getIPAddress = function() {
  $.getJSON('//ipinfo.io/json', function(data) {
    this.ipAddress = data;
  });
};

Sat.prototype.newItem = function(url) {
  let item = new this.ItemType(this, this.items.length, url);
  this.items.append(item);
  return item;
};

Sat.prototype.newLabelId = function() {
  let newId = this.lastLabelId + 1;
  this.lastLabelId = newId;
  return newId;
};

Sat.prototype.newLabel = function() {
  let label = new this.LabelType(this.currentItem, this.newLabelId());
  this.labelIdMap[label.id] = label;
  this.labels.append(label);
  this.currentItem.labels.append(label);
  return label;
};

Sat.prototype.addEvent = function(action, itemIndex, labelId = -1,
                                  position = null) {
  this.events.push({
    timestamp: Date.now(),
    action: action,
    itemIndex: itemIndex,
    labelId: labelId,
    position: position,
  });
};

// TODO
Sat.prototype.load = function () {

};

// TODO
Sat.prototype.submit = function() {

};

// TODO
Sat.prototype.gotoItem = function(index) {

};

/**
 * Information used for submission
 * @return {{items: Array, labels: Array, events: *, userAgent: string}}
 */
Sat.prototype.getInfo = function() {
  let self = this;
  let items = [];
  for (let i = 0; i < this.items.length; i++) {
    items.push(this.items[i].toJSON());
  }
  let labels = [];
  for (let i = 0; i < this.labels.length; i++) {
    labels.push(this.labels[i].toJson());
  }
  return {
    startTime: self.startTime,
    items: items,
    labels: labels,
    events: self.events,
    userAgent: navigator.userAgent,
    ipAddress: self.ipAddress,
  };
};

/**
 * Base class for each labeling target, can be pointcloud or 2D image
 * @param {Sat} sat: context
 * @param {number} index: index of this item in sat
 * @param {string | null} url: url to load the item
 */
function SatItem(sat, index = -1, url = null) {
  this.sat = sat;
  this.index = index;
  this.url = url;
  this.labels = [];
  this.ready = false;
}

SatItem.prototype.loaded = function() {
  this.ready = true;
  this.sat.addEvent('loaded', this.index);
};

SatItem.prototype.previousItem = function() {
  if (this.index === 0) {
    return null;
  }
  return this.sat.items[this.index-1];
};

SatItem.prototype.nextItem = function() {
  if (this.index < this.sat.items.length - 1) {
    return null;
  }
  return this.sat.items[this.index+1];
};

SatItem.prototype.toJson = function() {
  return {url: this.url, index: this.index};
};

SatItem.prototype.fromJson = function(object) {
  this.url = object.url;
  this.index = object.index;
};

SatItem.prototype.getVisibleLabels = function() {
  let labels = [];
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].valid && this.labels[i].numChildren === 0) {
      labels.push(this.labels[i]);
    }
  }
  return labels;
};

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
 * @param {Sat} sat: context
 * @param {number} index: index of this item in sat
 * @param {string} url: url to load the item
 */
function SatImage(sat, index, url) {
  SatItem.call(this, sat, index, url);
  this.image = new Image();
  this.image.onload = function() {
    this.loaded();
  };
  this.image.src = this.url;

  this.imageRatio = 0;

}

SatImage.prototype = Object.create(SatItem.prototype);

SatImage.prototype.loaded = function() {
  // Call SatItem loaded
  Object.getPrototypeOf(SatItem.prototype).loaded();
  // Show the image here when the image is loaded.
};

// TODO
SatImage.prototype.redraw = function() {

};

/**
 * Base class for all the labeled objects. New label should be instantiated by
 * Sat.newLabel()
 *
 * To define a new tool:
 *
 * function NewObject(id) {
 *   SatLabel.call(this, id);
 * }
 *
 * NewObject.prototype = Object.create(SatLabel.prototype);
 *
 * @param {SatItem} satItem: Item that this label appears
 * @param {number | null} id: label object identifier
 */
function SatLabel(satItem, id = -1) {
  this.id = id;
  this.name = null; // category or something else
  this.attributes = [];
  this.satItem = satItem;
  this.parent = null;
  this.children = [];
  this.numChildren = 0;
  this.valid = true;
}

SatLabel.prototype.delete = function() {
  this.valid = false;
  if (this.parent !== null) {
    this.parent.numChildren -= 1;
    if (this.parent.numChildren === 0) this.parent.delete();
  }
  for (let i = 0; i < this.children; i++) {
    this.children[i].parent = null;
    this.children[i].delete();
  }
};

SatLabel.prototype.getRoot = function() {
  if (this.parent === null) return this;
  else return this.parent.getRoot();
};

SatLabel.prototype.addChild = function(child) {
  this.numChildren += 1;
  this.children.push(child);
};

/**
 * Pick a color based on the label id
 * @return {(number|number|number)[]}
 */
SatLabel.prototype.color = function() {
  return pickColorPalette(this.getRoot().id);
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

/**
 * Return json object encoding the label information
 * @return {{id: *}}
 */
SatLabel.prototype.toJson = function() {
  let object = {id: this.id, item: this.satItem.index, name: this.name,
                attributes: this.attributes};
  if (this.parent !== null) object['parent'] = this.parent.id;
  if (this.children.length > 0) {
    let childenIds = [];
    for (let i = 0; i < this.children.length; i++) {
      childenIds.push(this.children[i].id);
    }
    object['children'] = childenIds;
  }
  return object;
};

/**
 * Load label information from json object
 * @param {Object} object: object to parse
 */
SatLabel.prototype.fromJson = function(object) {
  this.id = object.id;
  this.name = object.name;
  this.attributes = object.attributes;
  let labelIdMap = this.satItem.sat.labelIdMap;
  if ('parent' in object) {
    this.parent = labelIdMap[object['parent']];
  }
  if ('children' in object) {
    let childrenIds = object['children'];
    for (let i = 0; i < childrenIds.length; i++) {
      this.addChild(labelIdMap[childrenIds[i]]);
    }
  }
};
