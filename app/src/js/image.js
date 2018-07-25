
/* global SatItem SatLabel hiddenStyleColor UP_RES_RATIO
mode rgb pickColorPalette */
/* exported SatImage ImageLabel */

// constants
const DOUBLE_CLICK_WAIT_TIME = 300;
const CANVAS_STYLE_WIDTH = 900;
const CANVAS_STYLE_HEIGHT = 470;

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

/**
 * remove duplicate items in hidden map
 */
HiddenMap.prototype.removeDuplicate = function() {
  this.list = Array.from(new Set(this.list));
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
    if (label.valid) {
      shapes = shapes.concat(label.defaultHiddenShapes());
    }
  }
  this.resetHiddenMap(shapes);
};

SatImage.prototype._deselectAll = function() {
  if (this.selectedLabel) {
    this.selectedLabel.releaseAsTargeted();
    if (!this.selectedLabel.shapesValid()) {
      this.selectedLabel.delete();
    }
    this.selectedLabel = null;
  }
  if (this.active) {
    this.resetHiddenMapToDefault();
    this.redraw();
  }
};

SatImage.prototype.deselectAll = function() {
  for (let satImage of this.sat.items) {
    satImage._deselectAll();
  }
};

SatImage.prototype.deleteLabel = function(label) {
  if (label.parent) {
    label.parent.delete();
  } else {
    label.delete();
  }
};

SatImage.prototype._selectLabel = function(label) {
  if (this.selectedLabel) {
    this.selectedLabel.releaseAsTargeted();
    this.deselectAll();
  }

  this.selectedLabel = label;
  this.selectedLabel.setAsTargeted();

  for (let i = 0; i < this.sat.attributes.length; i++) {
    if (this.sat.attributes[i].toolType === 'switch') {
      this._setAttribute(i,
          this.selectedLabel.attributes[this.sat.attributes[i].name]);
    } else if (this.sat.attributes[i].toolType === 'list' &&
        this.sat.attributes[i].name in this.selectedLabel.attributes) {
      this._selectAttributeFromList(i,
          this.selectedLabel.attributes[this.sat.attributes[i].name][0]);
    }
  }
  if (this.active) {
    this._setCatSel(this.selectedLabel.categoryPath);
    this.redraw();
  }
};

SatImage.prototype.selectLabel = function(label) {
  // if the label has a parent, select all labels along the track
  if (label.parent) {
    let childIndex = label.parent.children.indexOf(label);
    let frameIndex = this.sat.items.indexOf(this);
    for (let i = 0; i < label.parent.children.length; i++) {
      let l = label.parent.children[i];
      this.sat.items[frameIndex + i - childIndex]._selectLabel(l);
    }
  } else {
    this._selectLabel(label);
  }
};

SatImage.prototype.updateLabelCount = function() {
  let numLabels = 0;
  for (let label of this.labels) {
    if (label.valid) {
      numLabels += 1;
    }
  }
  document.getElementById('label_count').textContent = '' + numLabels;
};

/**
 * Convert image coordinate to canvas coordinate.
 * If affine, assumes values to be [x, y]. Otherwise
 * performs linear transformation.
 * @param {[number]} values - the values to convert.
 * @param {boolean} affine - whether or not this transformation is affine.
 * @return {[number]} - the converted values.
 */
SatImage.prototype.toCanvasCoords = function(values, affine=true) {
  if (values) {
    for (let i = 0; i < values.length; i++) {
      values[i] *= this.displayToImageRatio;
    }
  }
  if (affine) {
    if (!this.padBox) {
      this.padBox = this._getPadding();
    }
    values[0] += this.padBox.x;
    values[1] += this.padBox.y;
  }
  return values;
};

/**
 * Convert canvas coordinate to image coordinate.
 * If affine, assumes values to be [x, y]. Otherwise
 * performs linear transformation.
 * @param {[number]} values - the values to convert.
 * @param {boolean} affine - whether or not this transformation is affine.
 * @return {[number]} - the converted values.
 */
SatImage.prototype.toImageCoords = function(values, affine=true) {
  if (affine) {
    if (!this.padBox) {
      this.padBox = this._getPadding();
    }
    values[0] -= this.padBox.x;
    values[1] -= this.padBox.y;
  }
  if (values) {
    for (let i = 0; i < values.length; i++) {
      values[i] /= this.displayToImageRatio;
    }
  }
  return values;
};

/**
 * Set the scale of the image in the display
 * @param {float} scale
 */
SatImage.prototype.setScale = function(scale) {
  let self = this;
  // set scale
  if (scale >= self.MIN_SCALE && scale < self.MAX_SCALE) {
    let ratio = scale / self.scale;
    self.mainCtx.scale(ratio, ratio);
    self.hiddenCtx.scale(ratio, ratio);
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
  self.imageCanvas.style.height =
      Math.round(CANVAS_STYLE_HEIGHT * self.scale) + 'px';
  self.imageCanvas.style.width =
      Math.round(CANVAS_STYLE_WIDTH * self.scale) + 'px';
  self.hiddenCanvas.style.height =
      Math.round(CANVAS_STYLE_HEIGHT * self.scale) + 'px';
  self.hiddenCanvas.style.width =
      Math.round(CANVAS_STYLE_WIDTH * self.scale) + 'px';

  self.imageCanvas.height =
      Math.round(CANVAS_STYLE_HEIGHT * UP_RES_RATIO * self.scale);
  self.imageCanvas.width =
      Math.round(CANVAS_STYLE_WIDTH * UP_RES_RATIO * self.scale);
  self.hiddenCanvas.height =
      Math.round(CANVAS_STYLE_HEIGHT * UP_RES_RATIO * self.scale);
  self.hiddenCanvas.width =
      Math.round(CANVAS_STYLE_WIDTH * UP_RES_RATIO * self.scale);
};

SatImage.prototype.loaded = function() {
  // Call SatItem loaded
  SatItem.prototype.loaded.call(this);
  if (this.active) {
    this.redraw();
  }
};

/**
 * Set whether this SatImage is the active one in the sat instance.
 * @param {boolean} active: if this SatImage is active
 */
SatImage.prototype.setActive = function(active) {
  SatItem.prototype.setActive.call(this);
  let self = this;
  self.active = active;
  let deleteBtn = $('#delete_btn');
  let endBtn = $('#end_btn');
  if (active) {
    self.lastLabelID = -1;
    self.padBox = self._getPadding();
    for (let i = 0; i < self.sat.items.length; i++) {
      self.sat.items[i].padBox = self.padBox;
    }

    self.imageCanvas.style.width = CANVAS_STYLE_WIDTH + 'px';
    self.imageCanvas.style.height = CANVAS_STYLE_HEIGHT + 'px';
    self.hiddenCanvas.style.width = CANVAS_STYLE_WIDTH + 'px';
    self.hiddenCanvas.style.height = CANVAS_STYLE_HEIGHT + 'px';
    self.mainCtx = self.imageCanvas.getContext('2d');
    self.hiddenCtx = self.hiddenCanvas.getContext('2d');

    self.imageCanvas.width = CANVAS_STYLE_WIDTH * UP_RES_RATIO;
    self.imageCanvas.height = CANVAS_STYLE_HEIGHT * UP_RES_RATIO;
    self.hiddenCanvas.width = CANVAS_STYLE_WIDTH * UP_RES_RATIO;
    self.hiddenCanvas.height = CANVAS_STYLE_HEIGHT * UP_RES_RATIO;

    self.mainCtx.scale(UP_RES_RATIO, UP_RES_RATIO);
    self.hiddenCtx.scale(UP_RES_RATIO, UP_RES_RATIO);

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
          self.deleteLabel(self.selectedLabel);
          self.deselectAll();
          self.redraw();
        }
      });
    }

    // toolbox
    self.sat.appendCascadeCategories(self.sat.categories, 0);
    self.catSel = document.getElementById('category_select');
    for (let i = 0; i < self.sat.attributes.length; i++) {
      let attributeName = self.sat.attributes[i].name;
      if (self.sat.attributes[i].toolType === 'switch') {
        $('#custom_attribute_' + attributeName).on(
          'switchChange.bootstrapSwitch', function(e) {
            e.preventDefault();
            self._attributeSwitch(i);
            self.redraw();
        });
      } else if (self.sat.attributes[i].toolType === 'list') {
        for (let j = 0; j < self.sat.attributes[i].values.length; j++) {
          $('#custom_attributeselector_' + i + '-' + j).on('click',
            function(e) {
            e.preventDefault();
            self._attributeListSelect(i, j);
            self.redraw();
          });
        }
      }
    }

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
  }
  if (self.selectedLabel) {
    // refresh hidden map
    self.selectLabel(self.selectedLabel);
  } else {
    self.resetHiddenMapToDefault();
  }

  self.redraw();
  self.updateLabelCount();
};

/**
 * Returns the currently selected attributes.
 * @private
 * @return {object} - the currently selected attributes.
 */
SatImage.prototype._getSelectedAttributes = function() {
  let self = this;
  let attributes = {};
  for (let i = 0; i < self.sat.attributes.length; i++) {
    let attributeName = self.sat.attributes[i].name;
    if (self.sat.attributes[i].toolType === 'switch') {
      attributes[attributeName] = document.getElementById(
        'custom_attribute_' + attributeName).checked;
    } else if (self.sat.attributes[i].toolType === 'list') {
      for (let j = 0; j < self.sat.attributes[i].values.length; j++) {
        if ($('#custom_attributeselector_' + i + '-' + j).hasClass('active')) {
          attributes[attributeName] = [j, self.sat.attributes[i].values[j]];
          break;
        }
      }
    }
  }
  return attributes;
};

/**
 * Prev button handler
 */
SatImage.prototype._prevHandler = function() {
  let self = this;
  self.deselectAll();
  self.sat.gotoItem(self.index - 1);
};

/**
 * Next button handler
 */
SatImage.prototype._nextHandler = function() {
  let self = this;
  self.deselectAll();
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
  self.deleteInvalidLabels();
  if (self.selectedLabel && !self.selectedLabel.valid) {
    self.selectedLabel = null;
  }
  // update the padding box
  self.padBox = self._getPadding();
  // draw stuff
  self.mainCtx.clearRect(0, 0, self.padBox.w * UP_RES_RATIO,
      self.padBox.h * UP_RES_RATIO);
  self.mainCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
      self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
  for (let label of self.labels) {
    if (label.valid) {
      label.redrawMainCanvas(self.mainCtx, self.hoveredLabel);
    }
  }
};

/**
 * Redraw the hidden canvas.
 */
SatImage.prototype.redrawHiddenCanvas = function() {
  let self = this;

  self.padBox = self._getPadding();
  self.hiddenCtx.clearRect(0, 0, self.padBox.w * UP_RES_RATIO,
      self.padBox.h * UP_RES_RATIO);
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
  self.mainCtx.clearRect(0, 0, self.padBox.w * UP_RES_RATIO,
      self.padBox.h * UP_RES_RATIO);
  for (let i = 0; i < self._hiddenMap.list.length; i++) {
    let shape = self._hiddenMap.get(i);
    shape.drawHidden(self.mainCtx, self, rgb(pickColorPalette(i)));
  }
};

/**
 * Checks if all existing labels are geometrically valid.
 * @return {boolean} whether all labels are geometrically valid.
 */
SatImage.prototype.shapesValid = function() {
  let shapesValid = true;
  for (let label of this.labels) {
    if (label.valid) {
      shapesValid = shapesValid && label.shapesValid();
    }
  }
  return shapesValid;
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
      self.deleteLabel(self.selectedLabel);
      self.deselectAll();
    }
  } else if (keyID === 188) { // +
    e.preventDefault();
    self._incHandler();
  } else if (keyID === 189) { // -
    e.preventDefault();
    self._decHandler();
  } else if (keyID === 37) { // Left/Right Arrow
    e.preventDefault();
    self._prevHandler();
  } else if (keyID === 39) { // Left/Right Arrow
    e.preventDefault();
    self._nextHandler();
  }
  self.redraw();
  self.updateLabelCount();
  if (keyID === 68) { // d for debug
    self.showHiddenCanvas();
  }
};

/**
 * Called when this SatImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousedown = function(e) {
  // do nothing if the user tries to click on the scroll bar
  if (e.offsetX > e.target.clientWidth || e.offsetY > e.target.clientHeight) {
    return;
  }

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
      self.catSel = document.getElementById('category_select');
      let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
      let attributes = self._getSelectedAttributes();
      self.selectLabel(self.sat.newLabel({
        categoryPath: cat, attributes: attributes, mousePos: mousePos,
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
  let rectDiv = this.divCanvas.getBoundingClientRect();
  let cH = $('#crosshair-h');
  let cV = $('#crosshair-v');
  cH.css('top', e.clientY);
  cH.css('left', rectDiv.x);
  cH.css('width', rectDiv.width);
  cV.css('left', e.clientX);
  cV.css('top', rectDiv.y);
  cV.css('height', rectDiv.height);
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

    // label specific handling of mousemove
    if (this.selectedLabel) {
      this.selectedLabel.mousemove(e);
    }

    // hover effect
    let hoveredShape = this.getOccupiedShape(mousePos);
    this.hoveredLabel = this.getLabelOfShape(hoveredShape);
    if (this.hoveredLabel) {
      this.hoveredLabel.setCurrHoveredShape(hoveredShape);
    }

    if (this.isMouseDown && this.selectedLabel) {
      this.imageCanvas.style.cursor = this.selectedLabel.getCursorStyle(
          this.selectedLabel.getSelectedShape());
    } else if (!this.isMouseDown && this.hoveredLabel) {
      this.imageCanvas.style.cursor = this.hoveredLabel.getCursorStyle(
          this.hoveredLabel.getCurrHoveredShape());
    }
  } else {
    if (this.selectedLabel) {
      this.selectedLabel.mouseleave(e);
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
  if (e.offsetX > e.target.clientWidth || e.offsetY > e.target.clientHeight) {
    return;
  }

  let self = this;
  if (!self._isWithinFrame(e)) {
    return;
  }

  if (this.sat.LabelType.useDoubleClick) {
    if (!self.selectedLabel && self.isMouseDown) {
      setTimeout(function() {
        if (!self.selectedLabel) {
          self.catSel = document.getElementById('category_select');
          let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
          let mousePos = self.getMousePos(e);

          let attributes = self._getSelectedAttributes();
          self.selectLabel(self.sat.newLabel({
                categoryPath: cat, attributes: attributes, mousePos: mousePos,
              })
          );
        }
      }, DOUBLE_CLICK_WAIT_TIME);
    } else if (self.selectedLabel) {
      self.selectedLabel.mouseup(e);
    }
  } else {
    if (self.selectedLabel) {
      self.selectedLabel.mouseup(e);
    }
  }
  if (!self.selectedLabel && self.sat.tracks) {
    // if tracking, propagate lack of label
    for (let i = 0; i < self.sat.items.length; i++) {
      self.sat.items[i].deselectAll();
    }
  }
  self.redraw();
  self.updateLabelCount();
  self.isMouseDown = false;
};

/**
 * True if mouse is within the image frame (tighter bound than canvas).
 * @param {object} e: mouse event
 * @return {boolean}: whether the mouse is within the image frame
 */
SatImage.prototype._isWithinFrame = function(e) {
  let rect = this.imageCanvas.getBoundingClientRect();
  let withinImage = (this.padBox
      && rect.x + this.padBox.x / UP_RES_RATIO < e.clientX
      && e.clientX <
        rect.x + this.padBox.x / UP_RES_RATIO + this.padBox.w / UP_RES_RATIO
      && rect.y + this.padBox.y / UP_RES_RATIO < e.clientY
      && e.clientY <
        rect.y + this.padBox.y / UP_RES_RATIO + this.padBox.h / UP_RES_RATIO);

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
  return {
    x: (e.clientX - rect.x - self.padBox.x / UP_RES_RATIO)
    / self.displayToImageRatio * UP_RES_RATIO,
    y: (e.clientY - rect.y - self.padBox.y / UP_RES_RATIO)
    / self.displayToImageRatio * UP_RES_RATIO};
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
    this.displayToImageRatio = this.imageCanvas.width / this.image.width;
    box.x = 0;
    box.y = 0.5 * (this.imageCanvas.height -
        this.image.height * this.displayToImageRatio);
    box.w = this.imageCanvas.width;
    box.h = this.imageCanvas.height - UP_RES_RATIO * box.y;
  } else {
    this.displayToImageRatio = this.imageCanvas.height / this.image.height;
    box.x = 0.5 * (this.imageCanvas.width -
        this.image.width * this.displayToImageRatio);
    box.y = 0;
    box.w = this.imageCanvas.width - UP_RES_RATIO * box.x;
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
  let [x, y] = this.toCanvasCoords([mousePos.x,
    mousePos.y]);
  let data = this.hiddenCtx.getImageData(x, y, 4, 4).data;
  let arr = [];
  for (let i = 0; i < 16; i++) {
    let color = (data[i*4] << 16) | (data[i*4+1] << 8) | data[i*4+2];
    arr.push(color);
  }
  // finding the mode of the data array to deal with anti-aliasing of the canvas
  return mode(arr) - 1;
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
    if (label.valid && label.selectedBy(shape)) {
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
SatImage.prototype._changeSelectedLabelCategory = function() {
  let self = this;
  if (self.selectedLabel) {
    self.catSel = document.getElementById('category_select');
    let option = self.catSel.options[self.catSel.selectedIndex].innerHTML;
    if (self.selectedLabel.parent) {
      for (let c of self.selectedLabel.parent.children) {
        c.categoryPath = option;
      }
    } else {
      self.selectedLabel.categoryPath = option;
    }
    self.redrawMainCanvas();
  }
};

/**
 * Called when an attribute checkbox is toggled.
 * @param {int} attributeIndex - the index of the attribute toggled.
 */
SatImage.prototype._attributeSwitch = function(attributeIndex) {
  let attributeName = this.sat.attributes[attributeIndex].name;
  if (this.selectedLabel) {
    if (this.selectedLabel.parent) {
      for (let l of this.selectedLabel.parent.children) {
        l.attributes[attributeName] = $('#custom_attribute_'
            + attributeName).prop('checked');
      }
    } else {
      this.selectedLabel.attributes[attributeName] = $('#custom_attribute_'
          + attributeName).prop('checked');
    }
  }
};

/**
 * Called when an attribute list is interacted with.
 * @param {int} attributeIndex - the index of the attribute interacted with.
 * @param {int} selectedIndex - the index of the selected value for the
 * attribute.
 */
SatImage.prototype._attributeListSelect = function(attributeIndex,
                                                   selectedIndex) {
  let attributeName = this.sat.attributes[attributeIndex].name;
  if (this.selectedLabel) {
    // store both the index and the value in order to prevent another loop
    //   during tag drawing
    this.selectedLabel.attributes[attributeName] =
      [selectedIndex,
        this.sat.attributes[attributeIndex].values[selectedIndex]];
    if (this.selectedLabel.parent) {
      this.selectedLabel.parent.interpolate(this.selectedLabel);
    }
  }
};

/**
 * Sets the value of a checkbox.
 * @param {int} attributeIndex - the index of the attribute toggled.
 * @param {boolean} value - the value to set.
 */
SatImage.prototype._setAttribute = function(attributeIndex, value) {
  let attributeName = this.sat.attributes[attributeIndex].name;
  let attributeCheckbox = $('#custom_attribute_' + attributeName);
  if (attributeCheckbox.prop('checked') !== value) {
    attributeCheckbox.trigger('click');
  }
  if (this.active) {
    this.redraw();
  }
};

/**
 * Sets the value of a list.
 * @param {int} attributeIndex - the index of the attribute toggled.
 * @param {int} selectedIndex - the index of the value selected.
 */
SatImage.prototype._selectAttributeFromList = function(attributeIndex,
                                                       selectedIndex) {
  let selector = $('#custom_attributeselector_' + attributeIndex + '-' +
    selectedIndex);
  if (!selector.hasClass('active')) {
    selector.trigger('click');
  }
};

/**
 * Used to set the value of the category selection index.
 * @param {number} categoryPath - the category path.
 */
SatImage.prototype._setCatSel = function(categoryPath) {
  this.catSel = document.getElementById('category_select');
  for (let i = 0; i < this.catSel.options.length; i++) {
    if (this.catSel.options[i].innerHTML === categoryPath) {
      this.catSel.selectedIndex = i;
      break;
    }
  }
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

  this.TAG_WIDTH = 25 * UP_RES_RATIO;
  this.TAG_HEIGHT = 14 * UP_RES_RATIO;
  // whether to draw this polygon in the targeted fill color
  this.targeted = false;
}

ImageLabel.prototype = Object.create(SatLabel.prototype);

ImageLabel.useCrossHair = false;
ImageLabel.defaultCursorStyle = 'auto';

ImageLabel.prototype.delete = function() {
  SatLabel.prototype.delete.call(this);
  this.deleteAllShapes();
};

ImageLabel.prototype.deleteAllShapes = function() {
  // specific to each class
};

ImageLabel.prototype.getCurrentPosition = function() {

};

ImageLabel.prototype.fromJsonPointers = function(json) {
  let self = this;
  self.decodeBaseJsonPointers(json);
  // self.satItem = self.sat.currentItem;
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
 * Set this label to have the provided shape.
 * @param {Shape} ignoredShape - The shape.
 */
ImageLabel.prototype.setShape = function(ignoredShape) {

};

/**
 * Shrink this label, forcing the user to update it.
 * @param {ImageLabel} ignoredStartLabel - The first label.
 */
ImageLabel.prototype.shrink = function(ignoredStartLabel) {

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
  if (self.shapesValid()) {
    ctx.save();
    let words = self.categoryPath.split(' ');
    let tw = self.TAG_WIDTH;
    // abbreviate tag as the first 3 chars of the last word
    let abbr = words[words.length - 1].substring(0, 3);
    for (let i = 0; i < self.sat.attributes.length; i++) {
      let attribute = self.sat.attributes[i];
      if (attribute.toolType === 'switch') {
        if (self.attributes[attribute.name]) {
          abbr+= ',' + attribute.tagText;
          tw += 18;
        }
      } else if (attribute.toolType === 'list') {
        if (self.attributes[attribute.name] &&
          self.attributes[attribute.name][0] > 0) {
          abbr += ',' + attribute.tagPrefix + ':' +
            attribute.tagSuffixes[self.attributes[attribute.name][0]];
          tw += 36;
        }
      }
    }

    let [tlx, tly] = self.satItem.toCanvasCoords(position);
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
