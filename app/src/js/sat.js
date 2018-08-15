
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
  self.load();
}

/**
 * Store IP information describing the user using the ipify service.
 */
Sat.prototype.getIpInfo = function() {
  $.getJSON('https://api.ipify.org?format=jsonp&callback=?', function(json) {
    this.ipInfo = json.ip;
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
 * @param {[float, float]} position - coordinate storing some representation of
 *  position at which this event occurred.
 */
Sat.prototype.addEvent = function(action, itemIndex, labelId = -1,
                                  position = null) {
  if (!this.events) {
    this.events = [];
  }
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
  if (self.slider) {
    self.slider.value = index + 1;
  }
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
  this.initToolbox();
  this.currentItem.setActive(true);
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
      if (self.demoMode) {
        document.getElementById('save_btn').style.display = 'none';
      }
      self.loaded();
    }
  };
  // get params from url path. These uniquely identify a SAT.
  let searchParams = new URLSearchParams(window.location.search);
  self.taskIndex = parseInt(searchParams.get('task_index'));
  self.projectName = searchParams.get('project_name');
  // send the request to the back end
  let request = JSON.stringify({'task': {
    'index': self.taskIndex,
    'projectOptions': {'name': self.projectName},
  }});
  xhr.open('POST', './postLoadAssignment', false);
  xhr.send(request);
};

// recursively create category to arbitrary level
Sat.prototype.appendCascadeCategories = function(
  subcategories, level, selectedIdx=0) {
  let self = this;
  // clean up
  let previousChildLevel = level;
  while (document.getElementById('parent_select_' + previousChildLevel)) {
    $('#parent_select_' + previousChildLevel).next().remove();
    document.getElementById('parent_select_' + previousChildLevel).remove();
    previousChildLevel++;
  }
  // base case null
  if (!subcategories) {return;}
  // get parent div
  let categoryDiv = document.getElementById('custom_categories');
  // build new category select window
  let child;
  // if there is subcategory, this node is parent node, grow tree
  if (subcategories[selectedIdx].subcategories) {
    child = document.createElement('select');
    child.id = 'parent_select_' + level;
    child.classList.add('form-control');
    child.size = Math.min(10, subcategories.length);
    child.style = 'font-size:15px';
  } else {
    // this node is leaf, clean up old options
    let oldCategorySelect = document.getElementById('category_select');
    if (oldCategorySelect) {
      oldCategorySelect.innerHTML = '';
      $('#category_select').next().remove();
      child = oldCategorySelect;
    } else {
      child = document.createElement('select');
      child.id = 'category_select';
      child.classList.add('form-control');
      child.style = 'font-size:15px';
    }
    child.size = Math.min(10, subcategories.length);
  }
  for (let subcategory of subcategories) {
    let option = document.createElement('option');
    option.innerHTML = subcategory.name;
    child.append(option);
  }
  child.selectedIndex = selectedIdx;
  categoryDiv.append(child);
  categoryDiv.append(document.createElement('hr')); // horizontal break
  // attach appropriate handler if not last level
  if (subcategories[selectedIdx].subcategories) {
    $('#parent_select_' + level).change(function() {
      let newSubcategories = self.categories;
      for (let i=0; i <= level; i++) {
        let idx = document.getElementById('parent_select_' + i).selectedIndex;
        newSubcategories = newSubcategories[idx].subcategories;
      }
      // handles the edge case where leaf categories are not at same level
      if (!newSubcategories) { // this level becomes the category_select level
        let thisLevel = document.getElementById('parent_select_' + level);
        $('#parent_select_' + level).next().remove();
        let categorySelect = document.getElementById('category_select');
        categorySelect.innerHTML = thisLevel.innerHTML;
        categorySelect.size = thisLevel.size;
        categorySelect.selectedIndex = thisLevel.selectedIndex;
        thisLevel.remove();
      }
      self.appendCascadeCategories(newSubcategories, level + 1);
      if (self.currentItem) {
        self.currentItem._changeSelectedLabelCategory();
      }
    });
  } else {
    $('#category_select').change(function() {
      let tempIdx = document.getElementById('category_select').selectedIndex;
      let level = 0;
      let newSubcategories = self.categories;
      while (document.getElementById('parent_select_' + level)) {
        let idx = document.getElementById('parent_select_' + level)
          .selectedIndex;
        newSubcategories = newSubcategories[idx].subcategories;
        level++;
      }
      if (newSubcategories[tempIdx].subcategories) {
        self.appendCascadeCategories(newSubcategories, level, tempIdx);
      }
      self.currentItem._changeSelectedLabelCategory();
    });
  }
  this.appendCascadeCategories(
    subcategories[selectedIdx].subcategories,
    level + 1); // recursively add new levels
};

Sat.prototype.initToolbox = function() {
  let self = this;
  // initialize all categories
  this.appendCascadeCategories(this.categories, 0);
  // initialize all the attribute selectors
  for (let i = 0; i < self.attributes.length; i++) {
    let attributeInput = document.getElementById('custom_attribute_' +
      self.attributes[i].name);
    if (self.attributes[i].toolType === 'switch') {
      attributeInput.type = 'checkbox';
      attributeInput.setAttribute('data-on-color', 'info');
      attributeInput.setAttribute('data-on-text', 'Yes');
      attributeInput.setAttribute('data-off-text', 'No');
      attributeInput.setAttribute('data-size', 'small');
      attributeInput.setAttribute('data-label-text', self.attributes[i].name);
      $('#custom_attribute_' + self.attributes[i].name).bootstrapSwitch();
    } else if (self.attributes[i].toolType === 'list') {
      let listOuterHtml = '<span>' + self.attributes[i].name + '</span>';
      listOuterHtml +=
        '<div id="radios" class="btn-group" data-toggle="buttons">';
      for (let j = 0; j < self.attributes[i].values.length; j++) {
        listOuterHtml +=
          '<button id="custom_attributeselector_' + i + '-' + j +
          '" class="btn btn-raised btn-' + self.attributes[i].buttonColors[j] +
          '"> <input type="radio"/>' + self.attributes[i].values[j] +
          '</button>';
      }
      attributeInput.outerHTML = listOuterHtml;
    } else {
      attributeInput.innerHTML = 'Error: invalid tool type "' +
        self.attributes[i].toolType + '"';
    }
  }
  document.getElementById('save_btn').onclick = function() {
    self.save();
  };
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
 * @return {object}
 */
Sat.prototype.toJson = function() {
  let self = this;
  return self.encodeBaseJson();
};

/**
 * Encode the base SAT objects. This should NOT be overloaded. Instead,
 * overload Sat.prototype.toJson()
 * @return {object} - JSON representation of the base functionality in this
 *   SAT.
 */
Sat.prototype.encodeBaseJson = function() {
  let self = this;
  let items = [];
  let labeledItemsCount = 0;
  for (let i = 0; i < self.items.length; i++) {
    items.push(self.items[i].toJson());
    if (self.items[i].labels.length > 0) {
      labeledItemsCount++;
    }
  }
  let labels = [];
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].valid) {
      labels.push(self.labels[i].toJson());
    }
  }
  return {
    task: {
      projectOptions: {
        name: self.projectName,
        itemType: self.itemType,
        labelType: self.labelType,
        taskSize: self.taskSize,
        handlerUrl: self.handlerUrl,
        pageTitle: self.pageTitle,
        categories: self.categories,
        attributes: self.attributes,
        labelImport: self.importFiles,
        instructions: self.instructions,
        demoMode: self.demoMode,
      },
      index: self.taskIndex,
      items: items,
    },
    workerId: self.workerId,
    labels: labels,
    events: self.events,
    startTime: self.startTime,
    numLabeledItems: labeledItemsCount,
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
  // import labels
  if (self.importFiles &&
      self.importFiles.length > 0) {
    self.importLabelsFromImportFiles();
    self.save();
  }
};

/**
 * Decode the base SAT objects. This should NOT be overloaded. Instead,
 * overload Sat.prototype.fromJson()
 * @param {string} json - The JSON representation.
 */
Sat.prototype.decodeBaseJson = function(json) {
  let self = this;
  self.projectName = json.task.projectOptions.name;
  self.itemType = json.task.projectOptions.itemType;
  self.labelType = json.task.projectOptions.labelType;
  self.taskSize = json.task.projectOptions.taskSize;
  self.handlerUrl = json.task.projectOptions.handlerUrl;
  self.pageTitle = json.task.projectOptions.pageTitle;
  self.instructions = json.task.projectOptions.instructions;
  if (self.instructions) {
    document.getElementById('instruction_btn').href = self.instructions;
  }
  self.demoMode = json.task.projectOptions.demoMode;
  self.categories = json.task.projectOptions.categories;
  self.attributes = json.task.projectOptions.attributes;
  self.importFiles = json.task.projectOptions.labelImport;
  self.taskIndex = json.task.index;
  for (let i = 0; json.labels && i < json.labels.length; i++) {
    // keep track of highest label ID
    self.lastLabelId = Math.max(self.lastLabelId, json.labels[i].id);
    let newLabel = new self.LabelType(self, json.labels[i].id);
    newLabel.fromJsonVariables(json.labels[i]);
    self.labelIdMap[newLabel.id] = newLabel;
    self.labels.push(newLabel);
  }
  for (let i = 0; i < json.task.items.length; i++) {
    let newItem = self.newItem(json.task.items[i].url);
    newItem.fromJson(json.task.items[i]);
  }
  self.workerId = json.workerId;
  self.events = json.events; // TODO: don't deserialize all events
  self.startTime = json.startTime;

  self.currentItem = self.items[0];

  for (let i = 0; json.labels && i < json.labels.length; i++) {
    self.labelIdMap[json.labels[i].id].fromJsonPointers(json.labels[i]);
  }
  self.addEvent('start labeling', self.currentItem.index);
};

Sat.prototype.importLabelsFromImportFiles = function() {
  let self = this;
  for (let i = 0; i < self.items.length; i++) {
    let item = self.items[i];
    for (let j = self.importFiles.length - 1; j >= 0; j--) {
      let importItem = self.importFiles[j];
      // correspondence by url
      if (importItem.url === item.url) {
        if (importItem.labels) {
          for (let labelToImport of importItem.labels) {
            self.lastLabelId += 1;
            let newLabel = new self.LabelType(self, self.lastLabelId);
            newLabel = newLabel.fromExportFormat(labelToImport);
            if (newLabel) {
              newLabel.satItem = self.items[i];
              self.labelIdMap[newLabel.id] = newLabel;
              self.labels.push(newLabel);
              self.items[i].labels.push(newLabel);
            }
          }
        }
        self.importFiles.splice(j, 1);
        break;
      }
    }
  }
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
 * Abstract function that should be implemented by child
 * See SatImage for example
 */
SatItem.prototype._changeSelectedLabelCategory = function() {};

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
 * @param {object} json - JSON representation of this SatItem.
 * @param {string} json.url - This SatItem's url.
 * @param {number} json.index - This SatItem's index in
 * @param {list} json.labelIds - The list of label ids of this SatItem's
 *   SatLabels.
 */
SatItem.prototype.fromJson = function(json) {
  let self = this;
  self.url = json.url;
  self.index = json.index;
  if (json.labelIds) {
    for (let i = 0; i < json.labelIds.length; i++) {
      let label = self.sat.labelIdMap[json.labelIds[i]];
      self.labels.push(label);
      label.satItem = this;
    }
  }
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
  this.interpolateHandler = null;
}

SatLabel.useCrossHair = false;


/**
 * Function to set the tool box of SatLabel.
 * @param {object} ignoredSatItem - the SatImage object.
 */
SatLabel.prototype.setToolBox = function(ignoredSatItem) {

};

/**
 * Set this label's category.
 * @param {string} categoryPath - The / delimited category path for this label.
 */
SatLabel.prototype.setCategoryPath = function(categoryPath) {
  let self = this;
  if (self.categoryPath === categoryPath) return;
  self.categoryPath = categoryPath;
  if (self.parent) {
    self.parent.setCategoryPath(categoryPath);
  }
  for (let i = 0; self.children && i < self.children.length; i++) {
    self.children[i].setCategoryPath(categoryPath);
  }
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

SatLabel.prototype.getRoot = function() {
  if (this.parent === null) return this;
  else return this.parent.getRoot();
};

SatLabel.prototype.addChild = function(child) {
  this.numChildren += 1;
  this.children.push(child);
  child.parent = this;
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
  self.attributes['isTrack'] = self.isTrack;
  let json = {id: self.id, categoryPath: self.categoryPath,
    attributes: self.attributes};
  if (self.parent) {
    json.parentId = self.parent.id;
  } else {
    json.parentId = -1;
  }
  if (self.children && self.children.length > 0) {
    let childrenIds = [];
    for (let i = 0; i < self.children.length; i++) {
      if (self.children[i].valid) {
        childrenIds.push(self.children[i].id);
      }
    }
    json.childrenIds = childrenIds;
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
  self.attributes = json.attributes;
  self.isTrack = json.attributes['isTrack'];
  // TODO: remove
  self.keyframe = json.keyframe;
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

SatLabel.prototype.redraw = function() {

};
