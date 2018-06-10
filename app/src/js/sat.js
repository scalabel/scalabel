
/* global module rgba */
/* exported Sat SatItem SatLabel */

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
  module.exports = SatLabel;
}

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
  let self = this;
  self.items = []; // a.k.a ImageList, but can be 3D model list
  self.labels = []; // list of label objects
  self.labelIdMap = {};
  self.lastLabelId = -1;
  self.currentItem = null;
  self.ItemType = ItemType;
  self.LabelType = LabelType;
  self.events = [];
  self.startTime = Date.now();
  self.taskId = null;
  self.projectName = null;
  self.ready = false;
  self.getIpInfo();
  if (self.slider) {
    self.numFrames = self.slider.max;
    self.slider.oninput = function() {
      self.moveSlider();
    };
  }
}

/**
 * Store IP information describing the user using the freegeoip service.
 */
Sat.prototype.getIpInfo = function() {
  let self = this;
  $.getJSON('http://freegeoip.net/json/?callback=?', function(data) {
    self.ipInfo = data;
  });
};

/**
 * Create a new item for this SAT.
 * @param {string} url - Location of the new item.
 * @return {SatItem} - The new item.
 */
Sat.prototype.newItem = function(url) {
  let self = this;
  let item = new self.ItemType(self, self.items.length, url);
  self.items.push(item);
  return item;
};

/**
 * Get a new label ID.
 * @return {int} - The new label ID.
 */
Sat.prototype.newLabelId = function() {
  let newId = this.lastLabelId + 1;
  while (newId in this.labelIdMap) {
    newId += 1;
  }
  this.lastLabelId = newId;
  return newId;
};

/**
 * Create a new label for this SAT.
 * @param {object} optionalAttributes - Optional attributes that may be used by
 *   subclasses of SatLabel.
 * @return {SatLabel} - The new label.
 */
Sat.prototype.newLabel = function(optionalAttributes) {
  let self = this;
  let label = new self.LabelType(self, self.newLabelId(), optionalAttributes);
  self.labelIdMap[label.id] = label;
  self.labels.push(label);
  if (self.currentItem) {
    self.currentItem.labels.push(label);
  }
  return label;
};

/**
 * Add an event to this SAT.
 * @param {string} action - The action triggering the event.
 * @param {int} itemIndex - Index of the item on which the event occurred.
 * @param {int} labelId - ID of the label pertaining to the event.
 * @param {object} position - Object storing some representation of position at
 *   which this event occurred.
 */
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

/**
 * Go to an item in this SAT, setting it to active.
 * @param {int} index - Index of the item to go to.
 */
Sat.prototype.gotoItem = function(index) {
  // mod the index to wrap around the list
  let self = this;
  index = (index + self.items.length) % self.items.length;
  // TODO: event?
  self.currentItem.setActive(false);
  self.currentItem = self.items[index];
  self.currentItem.setActive(true);
  self.currentItem.onload = function() {
    self.currentItem.redraw();
  };
  self.currentItem.redraw();
  self.slider.value = index + 1;
};

Sat.prototype.moveSlider = function() {
  let self = this;
  let oldItem = self.currentItem;
  self.currentItem = self.items[parseInt(self.slider.value) - 1];
  if (oldItem) {
    oldItem.setActive(false);
  }
  self.currentItem.setActive(true);
};

Sat.prototype.loaded = function() {
  this.ready = true;
};

/**
 * Load this SAT from the back end.
 */
Sat.prototype.load = function() {
  let self = this;
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      self.fromJson(json);
      self.loaded();
    }
  };
  // get params from url path. These uniquely identify a SAT.
  let searchParams = new URLSearchParams(window.location.search);
  self.taskIndex = parseInt(searchParams.get('task_index'));
  self.projectName = searchParams.get('project_name');
  // send the request to the back end
  let request = JSON.stringify({
    'index': self.taskIndex,
    'projectName': self.projectName,
  });
  xhr.open('POST', './postLoadTask', false);
  xhr.send(request);
};

/**
 * Save this labeling session to file by sending JSON to the back end.
 */
Sat.prototype.save = function() {
  let self = this;
  let json = self.toJson();
  let xhr = new XMLHttpRequest();
  xhr.open('POST', './postSave');
  xhr.send(JSON.stringify(json));
};

/**
 * Get this session's JSON representation
 * @return {{items: Array, labels: Array, events: *, userAgent: string}}
 */
Sat.prototype.toJson = function() {
  let self = this;
  return self.encodeBaseJson();
};

/**
 * Encode the base SAT objects. This should NOT be overloaded. Instead,
 * overload Sat.prototype.toJson()
 * @return {object} - JSON representation of the base functionality in this
 *   SAT. */
Sat.prototype.encodeBaseJson = function() {
  let self = this;
  let items = [];
  for (let i = 0; i < self.items.length; i++) {
    items.push(self.items[i].toJson());
  }
  let labels = [];
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].valid) {
      labels.push(self.labels[i].toJson());
    }
  }
  return {
    projectName: self.projectName,
    startTime: self.startTime,
    items: items,
    labels: labels,
    categories: self.categories,
    events: self.events,
    userAgent: navigator.userAgent,
    ipInfo: self.ipInfo,
  };
};

/**
 * Initialize this session from a JSON representation
 * @param {string} json - The JSON representation.
 */
Sat.prototype.fromJson = function(json) {
  let self = this;
  self.decodeBaseJson(json);
};

/**
 * Decode the base SAT objects. This should NOT be overloaded. Instead,
 * overload Sat.prototype.fromJson()
 * @param {string} json - The JSON representation.
 */
Sat.prototype.decodeBaseJson = function(json) {
  let self = this;
  for (let i = 0; json.labels && i < json.labels.length; i++) {
    // keep track of highest label ID
    self.lastLabelId = Math.max(self.lastLabelId, json.labels[i].id);
    let newLabel = new self.LabelType(self, json.labels[i].id);
    newLabel.fromJsonVariables(json.labels[i]);
    self.labelIdMap[newLabel.id] = newLabel;
    self.labels.push(newLabel);
  }

  for (let i = 0; i < json.items.length; i++) {
    let newItem = self.newItem(json.items[i].url);
    newItem.fromJson(json.items[i]);
  }

  self.categories = json.category;
  self.assignmentId = json.assignmentId;
  self.projectName = json.projectName;

  self.currentItem = self.items[0];
  self.currentItem.setActive(true);
  self.categories = json.categories;

  for (let i = 0; json.labels && i < json.labels.length; i++) {
    self.labelIdMap[json.labels[i].id].fromJsonPointers(json.labels[i]);
  }
  self.addEvent('start labeling', self.currentItem.index);
};


/**
 * Information used for submission
 * @return {{items: Array, labels: Array, events: *, userAgent: string}}
 */
Sat.prototype.getInfo = function() {
  let self = this;
  let items = [];
  for (let i = 0; i < this.items.length; i++) {
    items.push(this.items[i].toJson());
  }
  let labels = [];
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].valid) {
      labels.push(this.labels[i].toJson());
    }
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
 * @param {string} url: url to load the item
 */
function SatItem(sat, index = -1, url = '') {
  let self = this;
  self.sat = sat;
  self.index = index;
  self.url = url;
  self.labels = [];
  self.ready = false; // is this needed?
}

SatItem.prototype.setActive = function(active) {
  let self = this;
  if (active) {
    self.sat.addEvent('start labeling', self.index);
  } else {
    self.sat.addEvent('end labeling', self.index);
  }
};

/**
 * Called when this item is loaded.
 */
SatItem.prototype.loaded = function() {
  this.ready = true;
  this.sat.addEvent('loaded', this.index);
};

/**
 * Get the item before this one.
 * @return {SatItem} the item before this one
 */
SatItem.prototype.previousItem = function() {
  if (this.index === 0) {
    return null;
  }
  return this.sat.items[this.index-1];
};

/**
 * Get the SatItem after this one.
 * @return {SatItem} the item after this one
 */
SatItem.prototype.nextItem = function() {
  if (this.index + 1 >= this.sat.items.length) {
    return null;
  }
  return this.sat.items[this.index+1];
};

/**
 * Get this SatItem's JSON representation.
 * @return {object} JSON representation of this item
 */
SatItem.prototype.toJson = function() {
  let self = this;
  let labelIds = [];
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].valid) {
      labelIds.push(self.labels[i].id);
    }
  }
  return {url: self.url, index: self.index, labelIds: labelIds};
};

/**
 * Restore this SatItem from JSON.
 * @param {object} selfJson - JSON representation of this SatItem.
 * @param {string} selfJson.url - This SatItem's url.
 * @param {number} selfJson.index - This SatItem's index in
 * @param {list} selfJson.labelIds - The list of label ids of this SatItem's
 *   SatLabels.
 */
SatItem.prototype.fromJson = function(selfJson) {
  let self = this;
  self.url = selfJson.url;
  self.index = selfJson.index;
  if (selfJson.labelIds) {
    for (let i = 0; i < selfJson.labelIds.length; i++) {
      self.labels.push(self.sat.labelIdMap[selfJson.labelIds[i]]);
    }
  }
  self.attributes = selfJson.attributes;
};

/**
 * Get all the visible labels in this SatItem.
 * @return {Array} list of all visible labels in this SatItem
 */
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
 * Delete
 */
SatItem.prototype.deleteInvalidLabels = function() {
  let self = this;
  let valid = [];
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].valid) {
      valid.push(self.labels[i]);
    }
  }
  self.labels = valid;
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
 * @param {Sat} sat: The labeling session
 * @param {number | null} id: label object identifier
 * @param {object} ignored: ignored parameter for optional attributes.
 */
function SatLabel(sat, id = -1, ignored = null) {
  this.id = id;
  this.categoryPath = null;
  this.attributes = {};
  this.sat = sat;
  this.parent = null;
  this.children = [];
  this.numChildren = 0;
  this.valid = true;
  this.selectedShape = null;
  this.hoveredShape = null;
}

SatLabel.useCrossHair = false;


/**
 * Function to set the tool box of SatLabel.
 * @param {object} satItem - the SatImage object.
 */
SatLabel.setToolBox = function(satItem) { // eslint-disable-line

};

SatLabel.prototype.delete = function() {
  this.valid = false;
  for (let i = this.children.length - 1; i >= 0; i--) {
    this.children[i].parent = null;
    this.children[i].delete();
    this.children.pop();
  }
  if (this.parent) {
    this.parent.childDeleted();
  }
};

SatLabel.prototype.childDeleted = function() {
  this.numChildren -= 1;
  if (this.numChildren === 0) this.delete();
};

SatLabel.prototype.setSelectedShape = function(shape) {
  this.selectedShape = shape;
};

SatLabel.prototype.setHoveredShape = function(shape) {
  this.hoveredShape = shape;
};

SatLabel.prototype.getSelectedShape = function() {
  return this.selectedShape;
};

SatLabel.prototype.getHoveredShape = function() {
  return this.hoveredShape;
};

SatLabel.prototype.getRoot = function() {
  if (this.parent === null) return this;
  else return this.parent.getRoot();
};

/**
 * Get the current position of this label.
 */
SatLabel.prototype.getCurrentPosition = function() {

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
SatLabel.prototype.styleColor = function(alpha = 1.0) {
  return rgba(this.color(), alpha);
};

SatLabel.prototype.encodeBaseJson = function() {
  let self = this;
  let json = {id: self.id, categoryPath: self.categoryPath};
  if (self.parent) {
    json.parent = self.parent.id;
  } else {
    json.parent = -1;
  }
  if (self.children && self.children.length > 0) {
    let childrenIds = [];
    for (let i = 0; i < self.children.length; i++) {
      if (self.children[i].valid) {
        childrenIds.push(self.children[i].id);
      }
    }
    json.children = childrenIds;
  }
  json.previousLabelId = -1;
  json.nextLabelId = -1;
  if (self.previousLabelId) {
    json.previousLabelId = self.previousLabelId;
  }
  if (self.nextLabelId) {
    json.nextLabelId = self.nextLabelId;
  }
  // TODO: remove
  json.keyframe = self.keyframe;
  return json;
};

/**
 * Return json object encoding the label information
 * @return {{id: *}}
 */
SatLabel.prototype.toJson = function() {
  let self = this;
  return self.encodeBaseJson();
};

SatLabel.prototype.decodeBaseJsonVariables = function(json) {
  let self = this;
  self.id = json.id;
  self.categoryPath = json.categoryPath;
  // TODO: remove
  self.keyframe = json.keyframe;
  if (json.previousLabelId > -1) {
    self.previousLabelId = json.previousLabelId;
  }
  if (json.nextLabelId > -1) {
    self.nextLabelId = json.nextLabelId;
  }
};

SatLabel.prototype.decodeBaseJsonPointers = function(json) {
  let self = this;
  let labelIdMap = self.sat.labelIdMap;
  labelIdMap[self.id] = self;
  self.sat.lastLabelId = Math.max(self.sat.lastLabelId, self.id);
  if (json.parent > -1) {
    self.parent = labelIdMap[json.parent];
  }

  if (json.children) {
    let childrenIds = json.children;
    for (let i = 0; i < childrenIds.length; i++) {
      self.addChild(labelIdMap[childrenIds[i]]);
    }
  }
};

/**
 * Load label information from json object
 * @param {object} json: JSON representation of this SatLabel.
 */
SatLabel.prototype.fromJsonVariables = function(json) {
  let self = this;
  self.decodeBaseJsonVariables(json);
};

SatLabel.prototype.fromJsonPointers = function(json) {
  let self = this;
  self.decodeBaseJsonPointers(json);
};


SatLabel.prototype.startChange = function() {

};

SatLabel.prototype.updateChange = function() {

};

SatLabel.prototype.finishChange = function() {

};

SatLabel.prototype.redraw = function() {

};
