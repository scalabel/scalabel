
/* global SatItem SatLabel hiddenStyleColor */
/* exported SatImage ImageLabel */

// constants
const DOUBLE_CLICK_WAIT_TIME = 300;

/**
 * The data structure to aid the hidden canvas,
 * supports lookup from both the object and the index.
 */
function HiddenMap() {
  this.list = [];
}

/**
 * Append an object into the double map.
 * @param {Shape} ref: a shape to add into the hidden map.
 */
// TODO: need to check if added object already exists

HiddenMap.prototype.append = function(ref) {
  this.list.push(ref);
};

/**
 * Append a list of objects into the double map.
 * @param {[Shape]} shapes: a list of shapes to add into the hidden map.
 */
// TODO: need to check if added object already exists

HiddenMap.prototype.appendList = function(shapes) {
  for (let shape of shapes) {
    this.append(shape);
  }
};

HiddenMap.prototype.get = function(index) {
  if (index >= 0 && index < this.list.length) {
    return this.list[index];
  }
  return null;
};

HiddenMap.prototype.clear = function() {
  this.list = [];
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

  self.divCanvas = document.getElementById('div_canvas');
  self.imageCanvas = document.getElementById('image_canvas');
  self.hiddenCanvas = document.getElementById('hidden_canvas');
  self.mainCtx = self.imageCanvas.getContext('2d');
  self.hiddenCtx = self.hiddenCanvas.getContext('2d');

  self.imageHeight = self.imageCanvas.height;
  self.imageWidth = self.imageCanvas.width;
  self.hoveredLabel = null;

  self.MAX_SCALE = 3.0;
  self.MIN_SCALE = 1.0;
  self.SCALE_RATIO = 1.5;

  self.isMouseDown = false;
  self._hiddenMap = new HiddenMap();
}

SatImage.prototype = Object.create(SatItem.prototype);

SatImage.prototype.resetHiddenMapToDefault = function() {
  let shapes = [];
  for (let label of this.labels) {
    shapes = shapes.concat(label.defaultHiddenShapes());
  }
  this.resetHiddenMap(shapes);
};

SatImage.prototype.deselectAll = function() {
  if (this.selectedLabel) {
    this.selectedLabel.releaseAsTargeted();
    if (!this.selectedLabel.isValid()) {
      this.selectedLabel.delete();
    }
    this.selectedLabel = null;
  }
  this.resetHiddenMapToDefault();
  this.redraw();
};

SatImage.prototype.selectLabel = function(label) {
  if (this.selectedLabel) {
    this.selectedLabel.releaseAsTargeted();
    this.deselectAll();
  }

  this.selectedLabel = label;
  this.selectedLabel.setAsTargeted();

  this._setOccl(this.selectedLabel.attributes.occl);
  this._setTrunc(this.selectedLabel.attributes.trunc);
  this._setCatSel(this.selectedLabel.categoryPath);
  this.redraw();
};

SatImage.prototype.transformPoints = function(points) {
  let self = this;
  if (points) {
    for (let i = 0; i < points.length; i++) {
      points[i] = points[i] * self.scale;
    }
  }
  return points;
};

/**
 * Set the scale of the image in the display
 * @param {float} scale
 */
SatImage.prototype.setScale = function(scale) {
  let self = this;
  // set scale
  if (scale >= self.MIN_SCALE && scale < self.MAX_SCALE) {
    self.scale = scale;
  } else {
    return;
  }
  // handle buttons
  if (self.scale >= self.MIN_SCALE * self.SCALE_RATIO) {
    $('#decrease_btn').attr('disabled', false);
  } else {
    $('#decrease_btn').attr('disabled', true);
  }
  if (self.scale <= self.MAX_SCALE / self.SCALE_RATIO) {
    $('#increase_btn').attr('disabled', false);
  } else {
    $('#increase_btn').attr('disabled', true);
  }
  // resize canvas
  self.imageCanvas.height = self.imageHeight * self.scale;
  self.imageCanvas.width = self.imageWidth * self.scale;
  self.hiddenCanvas.height = self.imageHeight * self.scale;
  self.hiddenCanvas.width = self.imageWidth * self.scale;
};

SatImage.prototype.loaded = function() {
  // Call SatItem loaded
  SatItem.prototype.loaded.call(this);
};

/**
 * Set whether this SatImage is the active one in the sat instance.
 * @param {boolean} active: if this SatImage is active
 */
SatImage.prototype.setActive = function(active) {
  SatItem.prototype.setActive.call(this);
  let self = this;
  let removeBtn = $('#remove_btn');
  let deleteBtn = $('#delete_btn');
  let endBtn = $('#end_btn');
  if (active) {
    self.lastLabelID = -1;
    self.padBox = self._getPadding();

    self.imageCanvas.height = self.imageHeight;
    self.imageCanvas.width = self.imageWidth;
    self.hiddenCanvas.height = self.imageHeight;
    self.hiddenCanvas.width = self.imageWidth;
    self.setScale(self.MIN_SCALE);

    // global listeners
    document.onkeydown = function(e) {
      self._keydown(e);
    };
    document.onmousedown = function(e) {
      self._mousedown(e);
    };
    document.onmouseup = function(e) {
      self._mouseup(e);
    };
    document.onmousemove = function(e) {
      self._mousemove(e);
    };
    document.onscroll = function(e) {
      self._scroll(e);
    };
    if (self.sat.LabelType.useDoubleClick) {
      document.ondblclick = function(e) {
        self._doubleclick(e);
      };
    }

    // buttons
    document.getElementById('prev_btn').onclick = function()
    {self._prevHandler();};
    document.getElementById('next_btn').onclick = function()
    {self._nextHandler();};

    if (document.getElementById('increase_btn')) {
      document.getElementById('increase_btn').onclick = function()
      {self._incHandler();};
    }

    if (document.getElementById('decrease_btn')) {
      document.getElementById('decrease_btn').onclick = function()
      {self._decHandler();};
    }

    if (endBtn.length) {
      // if the end button exists (we have a sequence) then hook it up
      endBtn.click(function() {
        if (self.selectedLabel) {
          self.selectedLabel.parent.endTrack(self.selectedLabel);
          self.redraw();
        }
      });
    }
    if (deleteBtn.length) {
      deleteBtn.click(function() {
        if (self.selectedLabel) {
          self.selectedLabel.delete();
          self.satItem.deselectAll();
          self.redraw();
        }
      });
    }
    if (removeBtn.length) {
      removeBtn.click(function() {
        if (self.selectedLabel) {
          self.selectedLabel.delete();
          self.satItem.deselectAll();
          self.redraw();
        }
      });
    }

    // toolbox
    self.catSel = document.getElementById('category_select');
    self.catSel.selectedIndex = 0;
    self.occlCheckbox = document.getElementById('occluded_checkbox');
    self.truncCheckbox = document.getElementById('truncated_checkbox');

    $('#category_select').change(function() {
      self._changeCat();
    });
    $('[name=\'occluded-checkbox\']').on('switchChange.bootstrapSwitch',
        function() {
          self._occlSwitch();
        });
    $('[name=\'truncated-checkbox\']').on('switchChange.bootstrapSwitch',
        function() {
          self._truncSwitch();
        });

    // TODO: Wenqi
    // traffic light color

    // class specific tool box
    self.sat.LabelType.setToolBox(self);

    self.lastLabelID = 0;
  } else {
    // .click just adds a function to a list of functions that get executed,
    // therefore we need to turn off the old functions
    if (endBtn.length) {
      endBtn.off();
    }
    if (deleteBtn.length) {
      deleteBtn.off();
    }
    if (removeBtn.length) {
      removeBtn.off();
    }
  }
  self.resetHiddenMapToDefault();
  self.redraw();
};


/**
 * Prev button handler
 */
SatImage.prototype._prevHandler = function() {
  let self = this;
  self.sat.gotoItem(self.index - 1);
};

/**
 * Next button handler
 */
SatImage.prototype._nextHandler = function() {
  let self = this;
  self.sat.gotoItem(self.index + 1);
};

/**
 * Increase button handler
 */
SatImage.prototype._incHandler = function() {
  let self = this;
  self.setScale(self.scale * self.SCALE_RATIO);
  self.redraw();
};

/**
 * Decrease button handler
 */
SatImage.prototype._decHandler = function() {
  let self = this;
  self.setScale(self.scale / self.SCALE_RATIO);
  self.redraw();
};

/**
 * Redraw
 */
SatImage.prototype.redraw = function() {
  let self = this;
  self.redrawMainCanvas();
  self.redrawHiddenCanvas();
};


/**
 * Redraw the image canvas.
 */
SatImage.prototype.redrawMainCanvas = function() {
  let self = this;
  // need to do some clean up at the beginning
  if (self.selectedLabel && !self.selectedLabel.valid) {
    self.selectedLabel = null;
  }
  self.deleteInvalidLabels();
  // update the padding box
  self.padBox = self._getPadding();
  // draw stuff
  self.mainCtx.clearRect(0, 0, self.imageCanvas.width,
      self.imageCanvas.height);
  self.mainCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
      self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
  for (let i = 0; i < self.labels.length; i++) {
    self.labels[i].redrawMainCanvas(
        self.mainCtx, self.hoveredLabel);
  }
};

/**
 * Redraw the hidden canvas.
 */
SatImage.prototype.redrawHiddenCanvas = function() {
  let self = this;
  self.padBox = self._getPadding();
  self.hiddenCtx.clearRect(0, 0, self.hiddenCanvas.width,
      self.hiddenCanvas.height);
  for (let i = 0; i < self._hiddenMap.list.length; i++) {
    let shape = self._hiddenMap.get(i);
    shape.drawHidden(self.hiddenCtx, self, hiddenStyleColor(i));
  }
};

/**
 * Show the hidden canvas on the main canvas (debug purpose).
 */
SatImage.prototype.showHiddenCanvas = function() {
  let self = this;
  self.padBox = self._getPadding();
  self.mainCtx.clearRect(0, 0, self.hiddenCanvas.width,
      self.hiddenCanvas.height);
  for (let i = 0; i < self._hiddenMap.list.length; i++) {
    let shape = self._hiddenMap.get(i);
    shape.drawHidden(self.mainCtx, self, hiddenStyleColor(i));
  }
};

/**
 * Checks if all labels are valid.
 * @return {boolean} whether all labels are valid.
 */
SatImage.prototype.isValid = function() {
  let isValid = true;
  for (let label of this.labels) {
    isValid = isValid && label.isValid();
  }
  return isValid;
};

/**
 * Key down handler.
 * @param {type} e: Description.
 */
SatImage.prototype._keydown = function(e) {
  let self = this;
  // class-specific handling of keydown event
  if (self.selectedLabel) {
    self.selectedLabel.keydown(e);
  }

  let keyID = e.KeyCode ? e.KeyCode : e.which;
  if (keyID === 27) { // Esc
    // deselect
    self.deselectAll();
  } else if (keyID === 46 || keyID === 8) { // Delete or Backspace
    if (self.selectedLabel) {
      self.selectedLabel.delete();
      self.deselectAll();
    }
  } else if (keyID === 38) {// up
    self._incHandler();
  } else if (keyID === 40) {// down
    self._decHandler();
  } else if (keyID === 37 || keyID === 39) { // Left/Right Arrow
    if (keyID === 37) { // Left Arrow
      self._prevHandler();
    } else if (keyID === 39) { // Right Arrow
      self._nextHandler();
    }
  }
  self.redraw();
};

/**
 * Called when this SatImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousedown = function(e) {
  let self = this;
  if (!self._isWithinFrame(e)) {
    return;
  }
  self.isMouseDown = true;
  let mousePos = self.getMousePos(e);
  if (this.sat.LabelType.useDoubleClick) {
    // if using double click, label created at mouseup
    if (self.selectedLabel) {
      // if there is a label selected, let it handle mousedown
      self.selectedLabel.mousedown(e);
    }
  } else {
    // else, label created at mousedown
    let occupiedShape = self.getOccupiedShape(mousePos);
    let occupiedLabel = self.getLabelOfShape(occupiedShape);
    if (occupiedLabel) {
      self.selectLabel(occupiedLabel);
      self.selectedLabel.setSelectedShape(occupiedShape);
      self.selectedLabel.mousedown(e);
    } else {
      let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
      let occl = self.occlCheckbox.checked;
      let trunc = self.truncCheckbox.checked;
      self.selectLabel(self.sat.newLabel({
        categoryPath: cat, occl: occl,
        trunc: trunc, mousePos: mousePos,
      }));

      self.selectedLabel.mousedown(e);
    }
  }
  self.redrawMainCanvas();
};

/**
 * Called when this SatImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
SatImage.prototype._doubleclick = function(e) {
  let self = this;
  if (!self._isWithinFrame(e)) {
    return;
  }
  if (self.selectedLabel) {
    self.selectedLabel.doubleclick(e);
  } else {
    let mousePos = self.getMousePos(e);
    let occupiedShape = self.getOccupiedShape(mousePos);
    let occupiedLabel = self.getLabelOfShape(occupiedShape);

    if (occupiedLabel) {
      occupiedLabel.setSelectedShape(occupiedShape);
      // label specific handling of mousedown
      occupiedLabel.doubleclick(e);
    }
  }

  self.redrawMainCanvas();
};

/**
 * Function to draw the crosshair
 * @param {object} e: mouse event
 */
SatImage.prototype.drawCrossHair = function(e) {
  let divRect = this.divCanvas.getBoundingClientRect();
  let cH = $('#crosshair-h');
  let cV = $('#crosshair-v');
  cH.css('top', e.clientY);
  cH.css('left', divRect.x);
  cH.css('width', divRect.width);
  cV.css('left', e.clientX);
  cV.css('top', divRect.y);
  cV.css('height', divRect.height);
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
  if (this.sat.LabelType.useCrossHair) {
    this.drawCrossHair(e);
  }
  if (this._isWithinFrame(e)) {
    let mousePos = this.getMousePos(e);
    this.imageCanvas.style.cursor = this.sat.LabelType.defaultCursorStyle;
    // hover effect
    let hoveredShape = this.getOccupiedShape(mousePos);
    this.hoveredLabel = this.getLabelOfShape(hoveredShape);
    if (this.hoveredLabel) {
      this.hoveredLabel.setCurrHoveredShape(hoveredShape);
    }
    // label specific handling of mousemove
    if (this.selectedLabel) {
      this.selectedLabel.mousemove(e);
    }

    if (this.isMouseDown && this.selectedLabel) {
      this.imageCanvas.style.cursor = this.selectedLabel.getCursorStyle(
          this.selectedLabel.getSelectedShape());
    } else if (!this.isMouseDown && this.hoveredLabel) {
      this.imageCanvas.style.cursor = this.hoveredLabel.getCursorStyle(
          this.hoveredLabel.getCurrHoveredShape());
    }
  }
  this.redrawMainCanvas();
  // this.showHiddenCanvas();
};

/**
 * Called when this SatImage is active and the mouse is moved.
 * @param {object} e: mouse event
 */
SatImage.prototype._scroll = function(e) {
  let self = this;
  if (self.sat.LabelType.useCrossHair) {
    self.drawCrossHair(e);
  }
  self.redrawMainCanvas();
};

/**
 * Called when this SatImage is active and the mouse is released.
 * @param {object} e: mouse event (unused)
 */
SatImage.prototype._mouseup = function(e) {
  let self = this;
  if (!self._isWithinFrame(e)) {
    return;
  }
  self.isMouseDown = false;

  if (this.sat.LabelType.useDoubleClick) {
    if (!self.selectedLabel) {
      setTimeout(function() {
        if (!self.selectedLabel) {
          let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
          let mousePos = self.getMousePos(e);

          let occl = self.occlCheckbox.checked;
          let trunc = self.truncCheckbox.checked;
          self.selectLabel(self.sat.newLabel({
                categoryPath: cat, occl: occl,
                trunc: trunc, mousePos: mousePos,
              })
          );
        }
      }, DOUBLE_CLICK_WAIT_TIME);
    } else {
      self.selectedLabel.mouseup(e);
    }
  } else {
    if (self.selectedLabel) {
      self.selectedLabel.mouseup(e);
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
  let withinImage = (this.padBox
      && rect.x + this.padBox.x < e.clientX
      && e.clientX < rect.x + this.padBox.x + this.padBox.w
      && rect.y + this.padBox.y < e.clientY
      && e.clientY < rect.y + this.padBox.y + this.padBox.h);

  let rectDiv = this.divCanvas.getBoundingClientRect();
  let withinDiv = (rectDiv.x < e.clientX
      && e.clientX < rectDiv.x + rectDiv.width
      && rectDiv.y < e.clientY
      && e.clientY < rectDiv.y + rectDiv.height);
  return withinImage && withinDiv;
};

/**
 * Get the mouse position on the canvas in the image coordinates.
 * @param {object} e: mouse event
 * @return {object}: mouse position (x,y) on the canvas
 */
SatImage.prototype.getMousePos = function(e) {
  let self = this;
  let rect = self.hiddenCanvas.getBoundingClientRect();
  return {x: (e.clientX - rect.x) / self.scale,
    y: (e.clientY - rect.y) / self.scale};
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
 * Get the label under the mouse.
 * @param {object} mousePos: position of the mouse
 * @return {int}: the selected label
 */
SatImage.prototype.getIndexOnHiddenMap = function(mousePos) {
  let [x, y] = this.transformPoints([mousePos.x,
    mousePos.y]);
  let pixelData = this.hiddenCtx.getImageData(x, y, 1, 1).data;
  let color = (pixelData[0] << 16) || (pixelData[1] << 8) || pixelData[2];
  return color - 1;
};

/**
 * Get a label that a given Shape object belongs to.
 * @param {Shape} shape: the Shape object.
 * @return {ImageLabel}: a label that a given Shape object belongs to.
 */
SatImage.prototype.getLabelOfShape = function(shape) {
  if (shape === null) {
    return null;
  }

  for (let label of this.labels) {
    if (label.selectedBy(shape)) {
      return label;
    }
  }
  return null;
};

/**
 * Get the label under the mouse.
 * @param {object} mousePos: position of the mouse
 * @return {Shape}: the occupied shape
 */
SatImage.prototype.getOccupiedShape = function(mousePos) {
  let labelIndex = this.getIndexOnHiddenMap(mousePos);
  return this._hiddenMap.get(labelIndex);
};

/**
 * Clear the hidden map.
 */
SatImage.prototype.clearHiddenMap = function() {
  this._hiddenMap.clear();
};

/**
 * Reset the hidden map with given objects.
 * @param {[Shape]} shapes - shapes to initialize the hidden map with.
 */
SatImage.prototype.resetHiddenMap = function(shapes) {
  this._hiddenMap.clear();
  this._hiddenMap.appendList(shapes);
};

/**
 * Reset the hidden map with given objects.
 * @param {[Shape]} shapes to initialize the hidden map with.
 */
SatImage.prototype.pushToHiddenMap = function(shapes) {
  if (shapes) {
    this._hiddenMap.appendList(shapes);
  }
};

/**
 * Called when the selected category is changed.
 */
SatImage.prototype._changeCat = function() {
  let self = this;
  if (self.selectedLabel) {
    let option = self.catSel.options[self.catSel.selectedIndex].innerHTML;
    self.selectedLabel.categoryPath = option;
    self.redraw();
  }
};

/**
 * Called when the occluded checkbox is toggled.
 */
SatImage.prototype._occlSwitch = function() {
  if (this.selectedLabel) {
    this.selectedLabel.attributes.occl =
        $('[name=\'occluded-checkbox\']').prop('checked');
  }
  this.redraw();
};

/**
 * Called when the truncated checkbox is toggled.
 */
SatImage.prototype._truncSwitch = function() {
  if (this.selectedLabel) {
    this.selectedLabel.attributes.trunc =
        $('[name=\'truncated-checkbox\']').prop('checked');
  }
  this.redraw();
};

/**
 * Used to set the value of the occlusion checkbox.
 * @param {boolean} value - the value to set.
 */
SatImage.prototype._setOccl = function(value) {
  let occludedCheckbox = $('[name=\'occluded-checkbox\']');
  if (occludedCheckbox.prop('checked') !==
      value) {
    occludedCheckbox.trigger('click');
  }
  this.redraw();
};

/**
 * Used to set the value of the truncation checkbox.
 * @param {boolean} value - the value to set.
 */
SatImage.prototype._setTrunc = function(value) {
  let truncatedCheckbox = $('[name=\'truncated-checkbox\']');
  if (truncatedCheckbox.prop('checked') !==
      value) {
    truncatedCheckbox.trigger('click');
  }
  this.redraw();
};

/**
 * Used to set the value of the category selection index.
 * @param {number} categoryPath - the category path.
 */
SatImage.prototype._setCatSel = function(categoryPath) {
  for (let i = 0; i < this.catSel.options.length; i++) {
    if (this.catSel.options[i].innerHTML === categoryPath) {
      this.catSel.selectedIndex = i;
      break;
    }
  }
};

 /**
 * Called when the traffic light color choice is changed.
 */
SatImage.prototype._lightSwitch = function() {
  // TODO: Wenqi
};

/**
 * Base class for all the image labels. New label should be instantiated by
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
 * @param {number} id: label object identifier
 * @param {object} optionalAttributes: Optional attributes for the SatLabel.
 */
function ImageLabel(sat, id, optionalAttributes = null) {
  SatLabel.call(this, sat, id, optionalAttributes);
  if (optionalAttributes && optionalAttributes.satItem) {
    this.satItem = optionalAttributes.satItem;
  } else if (sat.currentItem) {
    this.satItem = sat.currentItem;
  } else {
    this.satItem = sat.items[0];
  }

  this.TAG_WIDTH = 25;
  this.TAG_HEIGHT = 14;
  // whether to draw this polygon in the targeted fill color
  this.targeted = false;
}

ImageLabel.prototype = Object.create(SatLabel.prototype);

ImageLabel.useCrossHair = false;
ImageLabel.defaultCursorStyle = 'auto';

ImageLabel.prototype.getCurrentPosition = function() {

};

ImageLabel.prototype.fromJsonPointers = function(json) {
  let self = this;
  self.decodeBaseJsonPointers(json);
  self.satItem = self.sat.currentItem;
};

/**
 * Get the weighted average between this label and a provided label.
 * @param {ImageLabel} ignoredLabel - The other label.
 * @param {number} ignoredWeight - The weight, b/w 0 and 1, higher
 * corresponds to
 *   closer to the other label.
 * @return {object} - The label's position.
 */
ImageLabel.prototype.getWeightedAvg = function(ignoredLabel, ignoredWeight) {
  return null;
};

/**
 * Set this label to be the weighted average of the two provided labels.
 * @param {ImageLabel} ignoredStartLabel - The first label.
 * @param {ImageLabel} ignoredEndLabel - The second label.
 * @param {number} ignoredWeight - The weight, b/w 0 and 1, higher
 *   corresponds to closer to endLabel.
 */
ImageLabel.prototype.weightedAvg = function(ignoredStartLabel, ignoredEndLabel,
                                            ignoredWeight) {

};

/**
 * Calculate the intersection between this and another ImageLabel
 * @param {ImageLabel} ignoredLabel - The other image label.
 * @return {number} - The intersection between the two labels.
 */
ImageLabel.prototype.intersection = function(ignoredLabel) {
  return 0;
};

/**
 * Calculate the union between this and another ImageLabel
 * @param {ImageLabel} ignoredLabel - The other image label.
 * @return {number} - The union between the two labels.
 */
ImageLabel.prototype.union = function(ignoredLabel) {
  return 0;
};

ImageLabel.prototype.setAsTargeted = function() {
  this.targeted = true;
};

ImageLabel.prototype.releaseAsTargeted = function() {
  this.targeted = false;
};

ImageLabel.prototype.isTargeted = function() {
  return this.targeted;
};

ImageLabel.prototype.setCurrHoveredShape = function(shape) {
  this.hoveredShape = shape;
};

ImageLabel.prototype.getCurrHoveredShape = function() {
  return this.hoveredShape;
};

/**
 * Draw the label tag of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {[number]} position - the position to draw the tag.
 */
ImageLabel.prototype.drawTag = function(ctx, position) {
  let self = this;
  if (self.isValid()) {
    ctx.save();
    let words = self.categoryPath.split(' ');
    let tw = self.TAG_WIDTH;
    // abbreviate tag as the first 3 chars of the last word
    let abbr = words[words.length - 1].substring(0, 3);
    if (self.attributes.occl) {
      abbr += ',o';
      tw += 9;
    }
    if (self.attributes.trunc) {
      abbr += ',t';
      tw += 9;
    }

    let [tlx, tly] = self.satItem.transformPoints(position);
    ctx.fillStyle = self.styleColor();
    ctx.fillRect(tlx + 1, tly - self.TAG_HEIGHT, tw,
        self.TAG_HEIGHT);
    ctx.fillStyle = 'rgb(0,0,0)';
    ctx.fillText(abbr, tlx + 3, tly - 3);
    ctx.restore();
  }
};

/**
 * Returns the default Shape objects to be drawn on the hidden canvas.
 * @return {[Shape]} the list of Shape objects.
 */
ImageLabel.defaultHiddenShapes = function() {
  return null;
};

// event handlers
ImageLabel.prototype.mousedown = function(e) { // eslint-disable-line

};

ImageLabel.prototype.mouseup = function(e) { // eslint-disable-line

};

ImageLabel.prototype.mousemove = function(e) { // eslint-disable-line

};

ImageLabel.prototype.doubleclick = function(e) { // eslint-disable-line

};

ImageLabel.prototype.keydown = function(e) { // eslint-disable-line

};
