/* global sprintf */

/* exported Sat SatImage SatLabel ImageLabel */

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
  this.startTime = Date.now();
  this.taskId = null;
  this.projectName = null;
}

Sat.prototype.getIPAddress = function() {
  $.getJSON('//freegeoip.net/json/?callback=?', function(data) {
    this.ipAddress = data;
  });
};

Sat.prototype.newItem = function(url) {
  let item = new this.ItemType(this, this.items.length, url);
  this.items.push(item);
  return item;
};

Sat.prototype.newLabelId = function() {
  let newId = this.lastLabelId + 1;
  this.lastLabelId = newId;
  return newId;
};

Sat.prototype.newLabel = function(optionalAttributes) {
  let self = this;
  let label = new self.LabelType(self, self.newLabelId(), optionalAttributes);
  self.labelIdMap[label.id] = label;
  self.labels.push(label);
  self.currentItem.labels.push(label);
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
Sat.prototype.load = function() {
  let self = this;
  let x = new XMLHttpRequest();
  x.onreadystatechange = function() {
    if (x.readyState === 4) {
      let assignment = JSON.parse(x.response);
      let itemLocs = assignment.items;
      self.addEvent('start labeling', self.currentItem); // ??
      // preload items
      self.items = [];
      for (let i = 0; i < itemLocs.length; i++) {
        self.items.push(new self.ItemType(self, i, itemLocs[i].url));
      }
      self.currentItem = self.items[0];
      self.currentItem.setActive(true);
      self.currentItem.image.onload = function() {
        self.currentItem.redraw();
      };
    }
  };
  // get params from url path
  let searchParams = new URLSearchParams(window.location.search);
  self.taskId = searchParams.get('task_id');
  self.projectName = searchParams.get('project_name');

  // ?
  let request = JSON.stringify({
    'assignmentId': self.taskId,
    'projectName': self.projectName,
  });
  x.open('POST', './requestSubmission');
  x.send(request);
};

// TODO
Sat.prototype.submit = function() {

};

// TODO
Sat.prototype.gotoItem = function(index) {
  //  TODO: save
  // mod the index to wrap around the list
  index = (index + this.items.length) % this.items.length;
  // TODO: event?
  this.currentItem.setActive(false);
  this.currentItem = this.items[index];
  this.currentItem.setActive(true);
  this.currentItem.onload = function() {
    this.currentItem.redraw();
  };
  this.currentItem.redraw();
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
  if (this.index + 1 >= this.sat.items.length) {
    return null;
  }
  return this.sat.items[this.index+1];
};

SatItem.prototype.toJson = function() {
  let labelIds = [];
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].valid) {
      labelIds.push(this.labels[i].id);
    }
  }
  return {url: this.url, index: this.index, labels: labelIds};
};

SatItem.prototype.fromJson = function(object) {
  this.url = object.url;
  this.index = object.index;
  for (let i = 0; i < object.labelIds.length; i++) {
    this.labels.push(this.sat.labelIdMap[object.labelIds[i]]);
  }
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

SatItem.prototype.deleteLabelById = function(labelId, back = true) {
  // TODO: refactor this ugly code
  let self = this;
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].id === labelId) {
      let currentItem = self.previousItem();
      let currentLabel = self.labels[i].previousLabel;
      while (back && currentItem) {
        for (let j = 0; j < currentItem.labels.length; j++) {
          if (currentItem.labels[j].id === currentLabel.id) {
            currentItem.labels.splice(j, 1);
            if (currentItem.selectedLabel &&
              currentItem.selectedLabel.id === currentLabel.id) {
              currentItem.selectedLabel = null;
            }
          }
        }
        if (currentLabel) {
          currentLabel = currentLabel.previousLabel;
        }
        currentItem = currentItem.previousItem();
      }
      currentItem = self.nextItem();
      currentLabel = self.labels[i].nextLabel;
      while (currentItem) {
        for (let j = 0; j < currentItem.labels.length; j++) {
          if (currentItem.labels[j].id === currentLabel.id) {
            currentItem.labels.splice(j, 1);
            if (currentItem.selectedLabel &&
              currentItem.selectedLabel.id === currentLabel.id) {
              currentItem.selectedLabel = null;
            }
          }
        }
        if (currentLabel) {
          currentLabel = currentLabel.nextLabel;
        }
        currentItem = currentItem.nextItem();
      }
      self.labels.splice(i, 1);
      if (self.selectedLabel && self.selectedLabel.id === labelId) {
        self.selectedLabel = null;
      }
      return;
    }
  }
};

/**
 * Base class for each targeted labeling Image.
 *
 * To define a new tool:
 *
 * function NewTool() {
 *   SatImage.call(this, sat, index, url);
 * }
 *
 * NewTool.prototype = Object.create(SatImage.prototype);
 *
 * @param {Sat} sat: context
 * @param {number} index: index of this item in sat
 * @param {string} url: url to load the item
 */
function SatImage(sat, index, url) {
  let self = this;
  SatItem.call(self, sat, index, url);
  self.image = new Image();
  self.image.onload = function() {
    self.loaded();
  };
  self.image.src = self.url;
}

SatImage.prototype = Object.create(SatItem.prototype);

SatImage.prototype.loaded = function() {
  // Call SatItem loaded
  SatItem.prototype.loaded.call(this);
};

/**
 * Set whether this SatImage is the active one in the sat instance.
 * @param {boolean} active: if this SatImage is active
 */
SatImage.prototype.setActive = function(active) {
  let self = this;
  if (active) {
    // set up class-specific environment
    self.sat.LabelType.setEnvironment(self.sat);

    // global listeners
    document.onmousedown = function(e) {
      self._mousedown(e);
    };
    document.onmousemove = function(e) {
      self._mousemove(e);
    };
    document.onmouseup = function(e) {
      self._mouseup(e);
    };

    // buttons
    document.getElementById('prev_btn').onclick = function() {
      console.log('prev');
      self.sat.gotoItem(env.index - 1);
    };
    document.getElementById('next_btn').onclick = function() {
      console.log('next');
      self.sat.gotoItem(env.index + 1);
    };

    if ($('#end_btn').length) {
      // if the end button exists (we have a sequence) then hook it up
      $('#end_btn').click(function() {
        console.log('end');
        if (self.selectedLabel) {
          self.deleteLabelById(env.selectedLabel.id, false);
          self.redraw();
        }
      });
    }
    if ($('#delete_btn').length) {
      $('#delete_btn').click(function() {
        console.log('delete');
        if (self.selectedLabel) {
          self.deleteLabelById(self.selectedLabel.id);
          self.redraw();
        }
      });
    }
    if ($('#remove_btn').length) {
      $('#remove_btn').click(function() {
        if (self.selectedLabel) {
          self.deleteLabelById(self.selectedLabel.id);
          self.redraw();
        }
      });
    }
  } else {
    // .click just adds a function to a list of functions that get executed,
    // therefore we need to turn off the old functions
    if ($('#end_btn').length) {
      console.log('test');
      // $('#end_btn').off();
    }
    // if ($('#delete_btn').length) {
    //   $('#delete_btn').off();
    // }
    // if ($('#remove_btn').length) {
    //   $('#remove_btn').off();
    // }
  }

};
    
/**
 * Redraws this SatImage and all labels.
 */
SatImage.prototype.redraw = function() {
  let self = this;
  self.padBox = self._getPadding();
  self.mainCtx.clearRect(0, 0, self.imageCanvas.width,
    self.imageCanvas.height);
  self.hiddenCtx.clearRect(0, 0, self.hiddenCanvas.width,
    self.hiddenCanvas.height);
  self.mainCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
    self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
  for (let i = 0; i < self.labels.length; i++) {
    self.labels[i].redraw(self.mainCtx, self.hiddenCtx, self.selectedLabel,
      self.resizeID === self.labels[i].id, self.hoverLabel,
        self.hoverHandle, i);
  }
};

/**
 * Called when this SatImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousedown = function(e) {
  
  if (this._isWithinFrame(e)) {
    let mousePos = this._getMousePos(e);
    [this.selectedLabel, currHandle] = this._getSelected(mousePos);
    if (this.selectedLabel) {
      this.selectedLabel.setCurrHandle(currHandle);
    }
    // change checked traits on label selection
    if (this.selectedLabel) {
      // label specific handling of mousedown
      this.selectedLabel._mousedown(e);

    } else {
      // otherwise, new label
      this.selectedLabel = this.sat.newLabel({
        category: this.catSel.options[this.catSel.selectedIndex].innerHTML, 
        occl: this.occlCheckbox.checked,
        trunc: this.truncCheckbox.checked, 
        mousePos: mousePos
      });
    }

  }
  this.redraw();
};

/**
 * Function to draw the crosshair
 * @param {object} e: mouse event
 */
SatImage.prototype.drawCrossHair = function(e, canvRect, mousePos) {

  let cH = $('#crosshair-h');
  let cV = $('#crosshair-v');
  cH.css('top', Math.min(canvRect.y + this.padBox.y + this.padBox.h, Math.max(
    e.clientY, canvRect.y + this.padBox.y)));
  cH.css('left', canvRect.x + this.padBox.x);
  cH.css('width', this.padBox.w);
  cV.css('left', Math.min(canvRect.x + this.padBox.x + this.padBox.w, Math.max(
    e.clientX, canvRect.x + this.padBox.x)));
  cV.css('top', canvRect.y + this.padBox.y);
  cV.css('height', this.padBox.h);
  if (this._isWithinFrame(e)) {
    $('.hair').show();
  } else {
    $('.hair').hide();
  }
};

/**
 * Called when this SatImage is active and the mouse is moved.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousemove = function(e) {

  if (this._isWithinFrame(e)) {
    let canvRect = this.imageCanvas.getBoundingClientRect();
    let mousePos = this._getMousePos(e);

    if (this.sat.LabelType.useCrossHair) {
      this.drawCrossHair(e, canvRect, mousePos);
    }

    // needed for on-hover animations
    // this.hoverLabel = this._getSelected(mousePos);

    // label specific handling of mousemove
    if (this.selectedLabel) {
      this.selectedLabel._mousemove(e);
    }
    // hover effect

    [this.hoverLabel, this.hoverHandle] = this._getSelected(mousePos);
    if (this.hoverLabel && this.hoverLabel != this.selectedLabel) {
      this.hoverLabel.setCurrHandle(currHandle);
      if (this.hoverHandle == 0) {
        this.imageCanvas.style.cursor = 'move';
      }
    } else {

      if (!this.selectedLabel) {
        this.imageCanvas.style.cursor = 'crosshair';
      }
    }

    this.redraw();
  }

};  

/**
 * Called when this SatImage is active and the mouse is released.
 * @param {object} _: mouse event (unused)
 */
SatImage.prototype._mouseup = function(_) { // eslint-disable-line
  
  if (this.selectedLabel) {
    // label specific handling of mouseup
    this.selectedLabel._mouseup();
    if (this.selectedLabel.isSmall()) {
      this.deleteLabelById(this.selectedLabel.id);
    }
  }
  this.redraw();

};

/**
 * True if mouse is within the image frame (tighter bound than canvas).
 * @param {object} e: mouse event
 * @return {boolean}: whether the mouse is within the image frame
 */
SatImage.prototype._isWithinFrame = function(e) {
  let rect = this.imageCanvas.getBoundingClientRect();
  return (this.padBox && rect.x + this.padBox.x < e.clientX && e.clientX <
    rect.x + this.padBox.x + this.padBox.w && rect.y + this.padBox.y <
    e.clientY && e.clientY < rect.y + this.padBox.y + this.padBox.h);
};

/**
 * Get the mouse position on the canvas.
 * @param {object} e: mouse event
 * @return {object}: mouse position (x,y) on the canvas
 */
SatImage.prototype._getMousePos = function(e) {
  let rect = this.imageCanvas.getBoundingClientRect();
  return {x: e.clientX - rect.x, y: e.clientY - rect.y};
};

/**
 * Get the padding for the image given its size and canvas size.
 * @return {object}: padding box (x,y,w,h)
 */
SatImage.prototype._getPadding = function() {
  // which dim is bigger compared to canvas
  let xRatio = this.image.width / this.imageCanvas.width;
  let yRatio = this.image.height / this.imageCanvas.height;
  // use ratios to determine how to pad
  let box = {x: 0, y: 0, w: 0, h: 0};
  if (xRatio >= yRatio) {
    box.x = 0;
    box.y = 0.5 * (this.imageCanvas.height - this.imageCanvas.width *
      this.image.height / this.image.width);
    box.w = this.imageCanvas.width;
    box.h = this.imageCanvas.height - 2 * box.y;
  } else {
    box.x = 0.5 * (this.imageCanvas.width - this.imageCanvas.height *
      this.image.width / this.image.height);
    box.y = 0;
    box.w = this.imageCanvas.width - 2 * box.x;
    box.h = this.imageCanvas.height;
  }
  return box;
};

/**
 * Get the label with a given id.
 * @param {number} labelID: id of the sought label
 * @return {ImageLabel}: the sought label
 */
SatImage.prototype._getLabelByID = function(labelID) {
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].id === labelID) {
      return this.labels[i];
    }
  }
};

/**
 * Get the box and handle under the mouse.
 * @param {object} mousePos: canvas mouse position (x,y)
 * @return {[ImageLabel, number]}: the box and handle (0-9) under the mouse
 */
SatImage.prototype._getSelected = function(mousePos) {
  
  let pixelData = this.hiddenCtx.getImageData(mousePos.x,
    mousePos.y, 1, 1).data;
  let selectedLabelIndex = null;
  let currHandle = null;
  if (pixelData[3] !== 0) {
    selectedLabelIndex = pixelData[0] * 256 + pixelData[1];
    currHandle = pixelData[2] - 1;
  }

  let selectedLabel = this.labels[selectedLabelIndex];

  return [selectedLabel, currHandle];
};


/**
 * Called when the selected category is changed.
 */
SatImage.prototype._changeCat = function() {
  
  if (this.selectedLabel) {
    let option = this.catSel.options[this.catSel.selectedIndex].innerHTML;
    this.selectedLabel.name = option;
    this.redraw();
  }
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
  this.name = null; // category or something else
  this.attributes = {};
  this.sat = sat;
  this.parent = null;
  this.children = [];
  this.numChildren = 0;
  this.valid = true;
  this.useCrossHair = false;
  this.currHandle = 0;
}

SatLabel.prototype.delete = function() {
  this.valid = false;
  if (this.parent !== null) {
    this.parent.numChildren -= 1;
    if (this.parent.numChildren === 0) this.parent.delete();
  }
  for (let i = 0; i < this.children.length; i++) {
    this.children[i].parent = null;
    this.children[i].delete();
  }
};

SatLabel.prototype.setCurrHandle = function(handle) {
  this.currHandle = handle;
}

SatLabel.prototype.getRoot = function() {
  if (this.parent === null) return this;
  else return this.parent.getRoot();
};

/**
 * Get the current position of this label.
 */
SatLabel.prototype.getCurrentPosition = function() {
  return;
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
  let object = {id: this.id, name: this.name, attributes: this.attributes};
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

SatLabel.prototype.startChange = function() {
};

SatLabel.prototype.updateChange = function() {

};

SatLabel.prototype.finishChange = function() {

};

SatLabel.prototype.redraw = function() {

};

/**
 * Load label information from json object
 * @param {Object} object: object to parse
 */
SatLabel.prototype.fromJson = function(object) {
  this.id = object.id;
  this.name = object.name;
  this.attributes = object.attributes;
  let labelIdMap = this.sat.labelIdMap;
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


/**
 * Base class for all the labeled objects. New label should be instantiated by
 * Sat.newLabel()
 *
 * To define a new tool:
 *
 * function NewObject(sat, id) {
 *   ImageLabel.call(this, sat, id);
 * }
 *
 * NewObject.prototype = Object.create(ImageLabel.prototype);
 *
 * @param {Sat} sat: The labeling session
 * @param {number | null} id: label object identifier
 * @param {object} optionalAttributes: Optional attributes for the SatLabel.
 */
function ImageLabel(sat, id, optionalAttributes = null) {
  SatLabel.call(this, sat, id, optionalAttributes);
  this.env = sat.currentItem;
}

ImageLabel.prototype = Object.create(SatLabel.prototype);

/**
 * Draw a specified resize handle of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
ImageLabel.prototype.drawHandle = function(ctx, handleNo) {
  let self = this;
  ctx.save(); // save the canvas context settings
  let posHandle = self._getHandle(handleNo);
  if (self.isSmall()) {
    ctx.fillStyle = 'rgb(169, 169, 169)';
  } else {
    ctx.fillStyle = self.styleColor();
  }
  ctx.lineWidth = self.LINE_WIDTH;
  if (posHandle) {
    ctx.beginPath();
    ctx.arc(posHandle.x, posHandle.y, self.HANDLE_RADIUS, 0, 2 * Math.PI);
    ctx.fill();
    if (!self.isSmall()) {
      ctx.strokeStyle = 'white';
      ctx.lineWidth = self.OUTLINE_WIDTH;
      ctx.stroke();
    }
  }
  ctx.restore(); // restore the canvas to saved settings
};