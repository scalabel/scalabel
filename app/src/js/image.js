import {SatItem, SatLabel, pickColorPalette} from './sat';
import {hiddenStyleColor, mode, rgb} from './utils';
import {UP_RES_RATIO} from './shape';

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
export function SatImage(sat, index, url) {
  let self = this;
  SatItem.call(self, sat, index, url);

  self.image = new Image();
  self.image.onload = function() {
    self.loaded();
  };
  self.image.onerror = function() {
    alert('Image ' + self.url + ' was not found.');
  };
  self.image.src = self.url;

  self.divCanvas = document.getElementById('div_canvas');
  self.imageCanvas = document.getElementById('image_canvas');
  self.labelCanvas = document.getElementById('label_canvas');
  self.hiddenCanvas = document.getElementById('hidden_canvas');
  self.imageCtx = self.imageCanvas.getContext('2d');
  self.labelCtx = self.labelCanvas.getContext('2d');
  self.hiddenCtx = self.hiddenCanvas.getContext('2d');

  self.hoveredLabel = null;

  self.MAX_SCALE = 3.0;
  self.MIN_SCALE = 1.0;
  self.SCALE_RATIO = 1.05;
  self.scrollTimer = null;

  self.isMouseDown = false;
  self._hiddenMap = new HiddenMap();
  self._keyDownMap = {};
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
      if (this.selectedLabel.parent) {
        this.selectedLabel.parent.delete();
      }
      this.selectedLabel.delete();
    }
    this.selectedLabel = null;
  }
  if (this.active) {
    this.resetHiddenMapToDefault();
    this.redrawLabelCanvas();
    this.redrawHiddenCanvas();
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

  if (this.active) {
    for (let i = 0; i < this.sat.attributes.length; i++) {
      if (this.sat.attributes[i].toolType === 'switch') {
        this._setAttribute(i,
            this.selectedLabel.attributes[this.sat.attributes[i].name]);
      } else if (this.sat.attributes[i].toolType === 'list') {
        // list attributes defaults to 0
        let selectedIndex = 0;
        if (this.sat.attributes[i].name in this.selectedLabel.attributes) {
          selectedIndex =
              this.selectedLabel.attributes[this.sat.attributes[i].name][0];
        }
        this._selectAttributeFromList(i, selectedIndex);
      }
    }
    this._setCatSel(this.selectedLabel.categoryPath);
    this.redrawLabelCanvas();
    this.redrawHiddenCanvas();
  }
};

SatImage.prototype.selectLabel = function(label) {
  // if the label has a parent, select all labels along the track
  if (label.parent) {
    for (let l of label.parent.children) {
      l.satItem._selectLabel(l);
    }
  } else {
    this._selectLabel(label);
  }
};

SatImage.prototype.updateLabelCount = function() {
  let numLabels = 0;
  if (this.sat.tracks) {
    for (let track of this.sat.tracks) {
      if (track.valid) {
        numLabels += 1;
      }
    }
  } else {
    for (let label of this.labels) {
      if (label.valid) {
        numLabels += 1;
      }
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
SatImage.prototype.toCanvasCoords = function(values, affine = true) {
  if (values) {
    for (let i = 0; i < values.length; i++) {
      values[i] *= this.displayToImageRatio * UP_RES_RATIO;
    }
  }
  if (affine) {
    if (!this.padBox) {
      this.padBox = this._getPadding();
    }
    values[0] += this.padBox.x * UP_RES_RATIO;
    values[1] += this.padBox.y * UP_RES_RATIO;
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
SatImage.prototype.toImageCoords = function(values, affine = true) {
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
 * @param {number} scale
 */
SatImage.prototype.setScale = function(scale) {
  let self = this;
  // set scale
  if (scale >= self.MIN_SCALE && scale < self.MAX_SCALE) {
    let ratio = scale / self.scale;
    self.imageCtx.scale(ratio, ratio);
    self.labelCtx.scale(ratio, ratio);
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
  let rectDiv = this.divCanvas.getBoundingClientRect();
  self.imageCanvas.style.height =
      Math.round(rectDiv.height * self.scale) + 'px';
  self.imageCanvas.style.width =
      Math.round(rectDiv.width * self.scale) + 'px';
  self.labelCanvas.style.height =
      Math.round(rectDiv.height * self.scale) + 'px';
  self.labelCanvas.style.width =
      Math.round(rectDiv.width * self.scale) + 'px';
  self.hiddenCanvas.style.height =
      Math.round(rectDiv.height * self.scale) + 'px';
  self.hiddenCanvas.style.width =
      Math.round(rectDiv.width * self.scale) + 'px';

  self.imageCanvas.width = rectDiv.width * self.scale;
  self.imageCanvas.height = rectDiv.height * self.scale;
  self.hiddenCanvas.height =
      Math.round(rectDiv.height * UP_RES_RATIO * self.scale);
  self.hiddenCanvas.width =
      Math.round(rectDiv.width * UP_RES_RATIO * self.scale);
  self.labelCanvas.height =
      Math.round(rectDiv.height * UP_RES_RATIO * self.scale);
  self.labelCanvas.width =
      Math.round(rectDiv.width * UP_RES_RATIO * self.scale);
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
  let trackLinkBtn = $('#track_link_btn');
  if (active) {
    self.lastLabelID = -1;
    self.padBox = self._getPadding();
    for (let i = 0; i < self.sat.items.length; i++) {
      self.sat.items[i].padBox = self.padBox;
    }

    self.hiddenCtx.scale(UP_RES_RATIO, UP_RES_RATIO);
    self.labelCtx.scale(UP_RES_RATIO, UP_RES_RATIO);
    self.setScale(1.0);

    // global listeners
    document.onkeydown = function(e) {
      self._keydown(e);
    };
    document.onkeyup = function(e) {
      self._keyup(e);
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
    self.divCanvas.onwheel = function(e) {
      self._scroll(e);
    };
    document.getElementsByTagName('BODY')[0].onresize = function() {
      self.setScale(self.scale);
      self.redraw();
    };
    if (self.sat.LabelType.useDoubleClick) {
      document.ondblclick = function(e) {
        self._doubleclick(e);
      };
    }

    // buttons
    document.getElementById('prev_btn').onclick = function() {
      self._prevHandler();
    };
    document.getElementById('next_btn').onclick = function() {
      self._nextHandler();
    };

    if (document.getElementById('increase_btn')) {
      document.getElementById('increase_btn').onclick = function() {
        self._incHandler();
      };
    }

    if (document.getElementById('decrease_btn')) {
      document.getElementById('decrease_btn').onclick = function() {
        self._decHandler();
      };
    }

    if (endBtn.length) {
      // if the end button exists (we have a sequence) then hook it up
      endBtn.click(function() {
        if (self.selectedLabel) {
          self.selectedLabel.parent.endTrack(self.selectedLabel);
          self.redrawLabelCanvas();
          self.redrawHiddenCanvas();
        }
      });
    }
    if (deleteBtn.length) {
      deleteBtn.click(function() {
        if (self.selectedLabel) {
          self.deleteLabel(self.selectedLabel);
          self.deselectAll();
          self.redrawLabelCanvas();
          self.redrawHiddenCanvas();
        }
      });
    }
    if (trackLinkBtn.length) {
      document.getElementById('track_link_btn').onclick = function() {
        if (self.selectedLabel) {
          if (self.sat.linkingTrack != null) {
            self.sat.linkTracks();
          }
          self.sat.toggleTrackLink(self.selectedLabel.getRoot());
        } else if (self.sat.linkingTrack != null) {
          self.sat.linkTracks();
          self.sat.toggleTrackLink(self.sat.linkingTrack);
        } else {
          alert('Please select a track before clicking Track-Link.');
        }
      };
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
              self.redrawLabelCanvas();
              self.redrawHiddenCanvas();
            });
      } else if (self.sat.attributes[i].toolType === 'list') {
        for (let j = 0; j < self.sat.attributes[i].values.length; j++) {
          $('#custom_attributeselector_' + i + '-' + j).on('click',
              function(e) {
                e.preventDefault();
                self._attributeListSelect(i, j);
                self.redrawLabelCanvas();
                self.redrawHiddenCanvas();
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
    for (let i = 0; i < self.sat.attributes.length; i++) {
      let attributeName = self.sat.attributes[i].name;
      if (self.sat.attributes[i].toolType === 'switch') {
        $('#custom_attribute_' + attributeName).off(
            'switchChange.bootstrapSwitch');
      } else if (self.sat.attributes[i].toolType === 'list') {
        for (let j = 0; j < self.sat.attributes[i].values.length; j++) {
          $('#custom_attributeselector_' + i + '-' + j).off('click');
        }
      }
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
  if (self.selectedLabel) {
    if (!self.selectedLabel.allowsLeavingCurrentItem()) {
      return;
    }
    self.selectedLabel.deactivate();
  }
  self.sat.gotoItem(self.index - 1);
};

/**
 * Next button handler
 */
SatImage.prototype._nextHandler = function() {
  let self = this;
  if (self.selectedLabel) {
    if (!self.selectedLabel.allowsLeavingCurrentItem()) {
      return;
    }
    self.selectedLabel.deactivate();
  }
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
  self.redrawImageCanvas();
  self.redrawLabelCanvas();
  self.redrawHiddenCanvas();
};

/**
 * Redraw the image canvas.
 */
SatImage.prototype.redrawImageCanvas = function() {
  let self = this;
  // update the padding box
  self.padBox = self._getPadding();
  // draw stuff
  self.imageCtx.clearRect(0, 0, self.padBox.w,
      self.padBox.h);
  self.imageCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
      self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
};

/**
 * Redraw the label canvas.
 */
SatImage.prototype.redrawLabelCanvas = function() {
  let self = this;
  // need to do some clean up at the beginning
  self.deleteInvalidLabels();
  if (self.selectedLabel && !self.selectedLabel.valid) {
    self.selectedLabel = null;
  }
  self.labelCtx.clearRect(0, 0,
      (2 * self.padBox.x + self.imageCanvas.width) * UP_RES_RATIO,
      (2 * self.padBox.y + self.imageCanvas.height) * UP_RES_RATIO);
  for (let label of self.labels) {
    if (label.valid) {
      label.redrawLabelCanvas(self.labelCtx, self.hoveredLabel);
    }
  }
};

/**
 * Redraw the hidden canvas.
 */
SatImage.prototype.redrawHiddenCanvas = function() {
  let self = this;

  self.padBox = self._getPadding();
  self.hiddenCtx.clearRect(0, 0, (self.padBox.x + self.padBox.w) * UP_RES_RATIO,
      (self.padBox.y + self.padBox.h) * UP_RES_RATIO);
  for (let i = 0; i < self._hiddenMap.list.length; i++) {
    let shape = self._hiddenMap.get(i);
    shape.drawHidden(self.hiddenCtx, self, hiddenStyleColor(i));
  }
};

/**
 * Show the hidden canvas on the label canvas (debug purpose).
 */
SatImage.prototype.showHiddenCanvas = function() {
  let self = this;
  self.padBox = self._getPadding();
  self.labelCtx.clearRect(0, 0, self.padBox.w, self.padBox.h);
  for (let i = 0; i < self._hiddenMap.list.length; i++) {
    let shape = self._hiddenMap.get(i);
    shape.drawHidden(self.labelCtx, self, rgb(pickColorPalette(i)));
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
  self._keyDownMap[keyID] = true;
  if (keyID === 27) { // Esc
    // deselect
    self.deselectAll();
  } else if (keyID === 46 || keyID === 8) { // Delete or Backspace
    if (self.selectedLabel) {
      self.deleteLabel(self.selectedLabel);
      self.deselectAll();
    }
  } else if (keyID === 37) { // Left/Right Arrow
    e.preventDefault();
    self._prevHandler();
    return;
  } else if (keyID === 39) { // Left/Right Arrow
    e.preventDefault();
    self._nextHandler();
    return;
  } else if (keyID === 17) { // Ctrl
    self.ctrlDown = true;
  } else if (keyID === 38) { // up key
    if (this.selectedLabel) {
      e.preventDefault();
      let index = this.labels.indexOf(this.selectedLabel);
      if (index < this.labels.length - 1) {
        this.labels[index] = this.labels[index + 1];
        this.labels[index + 1] = this.selectedLabel;
      }
    }
  } else if (keyID === 40) { // down key
    if (this.selectedLabel) {
      e.preventDefault();
      let index = this.labels.indexOf(this.selectedLabel);
      if (index > 0) {
        this.labels[index] = this.labels[index - 1];
        this.labels[index - 1] = this.selectedLabel;
      }
    }
  } else if (keyID === 70) { // f for front
    if (this.selectedLabel) {
      let index = this.labels.indexOf(this.selectedLabel);
      if (index < this.labels.length - 1) {
        this.labels.splice(index, 1);
        this.labels.push(this.selectedLabel);
      }
    }
  } else if (keyID === 66) { // b for back
    if (this.selectedLabel) {
      let index = this.labels.indexOf(this.selectedLabel);
      if (index > 0) {
        this.labels.splice(index, 1);
        this.labels.unshift(this.selectedLabel);
      }
    }
  } else if (keyID === 72) { // h for hiding all labels
    if (this.labelCanvas.style.visibility === 'visible') {
      this.labelCanvas.style.visibility = 'hidden';
    } else {
      this.labelCanvas.style.visibility = 'visible';
    }
  } else if (keyID === 187) {
    this._incHandler();
    return;
  } else if (keyID === 189) {
    this._decHandler();
    return;
  }
  this.redrawLabelCanvas();
  this.redrawHiddenCanvas();
  self.updateLabelCount();
  if (keyID === 220) { // backslash for showing hidden canvas (debug)
    self.showHiddenCanvas();
  }
};

SatImage.prototype._keyup = function(e) {
  let self = this;
  let keyID = e.KeyCode ? e.KeyCode : e.which;
  delete self._keyDownMap[keyID];
  if (self.selectedLabel) {
    self.selectedLabel.keyup(e);
  }
  if (keyID === 17) { // Ctrl
    self.ctrlDown = false;
  }
};

SatImage.prototype.isDown = function(c) {
  return this._keyDownMap[c.charCodeAt()];
};

SatImage.prototype.anyKeyDown = function(keys) {
  for (let key of keys) {
    if (this.isDown(key)) {
      return true;
    }
  }
  return false;
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
  // only applies to left click
  if (e.which !== 1) {
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
      if (this.sat.linkingTrack && occupiedLabel.getRoot().id !==
          this.sat.linkingTrack.id) {
        this.sat.addTrackToLinkingTrack(occupiedLabel.getRoot());
      } else {
        self.selectLabel(occupiedLabel);
        self.selectedLabel.setSelectedShape(occupiedShape);
        self.selectedLabel.mousedown(e);
      }
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
  self.redrawLabelCanvas();
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

  self.redrawLabelCanvas();
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
    // label specific handling of mousemove
    if (this.selectedLabel) {
      this.selectedLabel.mousemove(e);
    }

    // hover effect
    let hoveredShape = this.getOccupiedShape(mousePos);
    let hoveredLabel = this.getLabelOfShape(hoveredShape);
    // hovered label changed
    if (this.hoveredLabel && this.hoveredLabel !== hoveredLabel) {
      this.hoveredLabel.releaseCurrHoveredShape();
    }
    if (hoveredLabel) {
      this.hoveredLabel = hoveredLabel;
      this.hoveredLabel.setCurrHoveredShape(hoveredShape);
    }

    if (this.isMouseDown && this.selectedLabel) {
      this.labelCanvas.style.cursor = this.selectedLabel.getCursorStyle(
          this.selectedLabel.getSelectedShape());
    } else if (!this.isMouseDown && this.hoveredLabel) {
      this.labelCanvas.style.cursor = this.hoveredLabel.getCursorStyle(
          this.hoveredLabel.getCurrHoveredShape());
    } else {
      this.labelCanvas.style.cursor = this.sat.LabelType.defaultCursorStyle;
    }
  } else {
    if (this.selectedLabel) {
      this.selectedLabel.mouseleave(e);
    }
  }
  this.redrawLabelCanvas();
};

/**
 * Called when this SatImage is active and the mouse is moved.
 * @param {object} e: mouse event
 */
SatImage.prototype._scroll = function(e) {
  let self = this;
  if (self.ctrlDown) { // control for zoom
    e.preventDefault();
    if (self.scrollTimer !== null) {
      clearTimeout(self.scrollTimer);
    }
    if (e.deltaY < 0) {
      self.setScale(self.scale * self.SCALE_RATIO);
    } else if (e.deltaY > 0) {
      self.setScale(self.scale / self.SCALE_RATIO);
    }
    self.redrawImageCanvas();
    self.redrawLabelCanvas();
    self.scrollTimer = setTimeout(function() {
      self.redrawHiddenCanvas();
    }, 150);
    return;
  }
  if (self.sat.LabelType.useCrossHair) {
    self.drawCrossHair(e);
  }
};

/**
 * Called when this SatImage is active and the mouse is released.
 * @param {object} e: mouse event (unused)
 */
SatImage.prototype._mouseup = function(e) {
  if (e.offsetX > e.target.clientWidth || e.offsetY > e.target.clientHeight) {
    return;
  }
  // only applies to left click
  if (e.which !== 1) {
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
              }),
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
    self.deselectAll();
  }
  self.redrawLabelCanvas();
  self.redrawHiddenCanvas();
  self.updateLabelCount();
  self.isMouseDown = false;
};

/**
 * True if mouse is within the image frame (tighter bound than canvas).
 * @param {object} e: mouse event
 * @return {boolean}: whether the mouse is within the image frame
 */
SatImage.prototype._isWithinFrame = function(e) {
  let rectDiv = this.divCanvas.getBoundingClientRect();
  return (rectDiv.x - 10 < e.clientX
      && e.clientX < rectDiv.x + rectDiv.width + 10
      && rectDiv.y - 10 < e.clientY
      && e.clientY < rectDiv.y + rectDiv.height + 10);
};

/**
 * Get the mouse position on the canvas in the image coordinates.
 * @param {object} e: mouse event
 * @return {object}: mouse position (x,y) on the canvas
 */
SatImage.prototype.getMousePos = function(e) {
  let self = this;
  // limit mouse within the image
  let rect = self.hiddenCanvas.getBoundingClientRect();
  let x = Math.min(
      Math.max(e.clientX, rect.x + this.padBox.x),
      rect.x + this.padBox.x + this.padBox.w);
  let y = Math.min(
      Math.max(e.clientY, rect.y + this.padBox.y),
      rect.y + this.padBox.y + this.padBox.h);

  // limit mouse within the main div
  let rectDiv = this.divCanvas.getBoundingClientRect();
  x = Math.min(
      Math.max(x, rectDiv.x),
      rectDiv.x + rectDiv.width
  );
  y = Math.min(
      Math.max(y, rectDiv.y),
      rectDiv.y + rectDiv.height
  );
  return {
    x: (x - rect.x - self.padBox.x) / self.displayToImageRatio,
    y: (y - rect.y - self.padBox.y) / self.displayToImageRatio,
  };
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
    box.h = this.imageCanvas.height - 2 * box.y;
  } else {
    this.displayToImageRatio = this.imageCanvas.height / this.image.height;
    box.x = 0.5 * (this.imageCanvas.width -
        this.image.width * this.displayToImageRatio);
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
  let [x, y] = this.toCanvasCoords([
    mousePos.x,
    mousePos.y]);
  let data = this.hiddenCtx.getImageData(x, y, 4, 4).data;
  let arr = [];
  for (let i = 0; i < 16; i++) {
    let color = (data[i * 4] << 16) | (data[i * 4 + 1] << 8) | data[i * 4 + 2];
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
    self.selectedLabel.setCategoryPath(option);
    self.redrawLabelCanvas();
  }
};

/**
 * Called when an attribute checkbox is toggled.
 * @param {int} attributeIndex - the index of the attribute toggled.
 */
SatImage.prototype._attributeSwitch = function(attributeIndex) {
  let attributeName = this.sat.attributes[attributeIndex].name;
  if (this.selectedLabel) {
    let checked = $('#custom_attribute_' + attributeName).prop('checked');
    if (this.selectedLabel.parent) {
      this.selectedLabel.parent.childAttributeChanged(
          attributeName, checked, this.selectedLabel.id);
    }
    this.selectedLabel.attributes = {...this.selectedLabel.attributes};
    this.selectedLabel.attributeframe = true;
    this.selectedLabel.attributes[attributeName] = checked;
  }
};

/**
 * Called when an attribute list is interacted with.
 * @param {int} attributeIndex - the index of the attribute interacted with.
 * @param {int} selectedIndex - the index of the selected value for the
 * attribute.
 */
SatImage.prototype._attributeListSelect = function(
    attributeIndex,
    selectedIndex) {
  let attributeName = this.sat.attributes[attributeIndex].name;
  if (this.selectedLabel) {
    // store both the index and the value in order to prevent another loop
    //   during tag drawing
    let value = this.sat.attributes[attributeIndex].values[selectedIndex];
    if (this.selectedLabel.parent) {
      this.selectedLabel.parent.childAttributeChanged(attributeName,
        [selectedIndex, value], this.selectedLabel.id);
    }
    this.selectedLabel.attributes = {...this.selectedLabel.attributes};
    this.selectedLabel.attributeframe = true;
    this.selectedLabel.attributes[attributeName] =
        [selectedIndex, value];
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
    attributeCheckbox.prop('checked', value);
    attributeCheckbox.bootstrapSwitch('state', value);
  }
  if (this.active) {
    this.redrawLabelCanvas();
  }
};

/**
 * Sets the value of a list.
 * @param {int} attributeIndex - the index of the attribute toggled.
 * @param {int} selectedIndex - the index of the value selected.
 */
SatImage.prototype._selectAttributeFromList = function(
    attributeIndex,
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
export function ImageLabel(sat, id, optionalAttributes = null) {
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
ImageLabel.allowsLinkingWithinFrame = false;

ImageLabel.prototype.delete = function() {
  SatLabel.prototype.delete.call(this);
  this.deleteAllShapes();
};

ImageLabel.prototype.deleteAllShapes = function() {
  // specific to each class
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

ImageLabel.prototype.getCurrentPosition = function() {

};

ImageLabel.prototype.setSelectedShape = function(shape) {
  this.selectedShape = shape;
};

ImageLabel.prototype.setHoveredShape = function(shape) {
  this.hoveredShape = shape;
};

ImageLabel.prototype.getSelectedShape = function() {
  return this.selectedShape;
};

ImageLabel.prototype.getHoveredShape = function() {
  return this.hoveredShape;
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

ImageLabel.prototype.releaseCurrHoveredShape = function() {
  this.hoveredShape = null;
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
          abbr += ',' + attribute.tagText;
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
    ctx.fillRect(tlx + UP_RES_RATIO, tly - self.TAG_HEIGHT * UP_RES_RATIO,
        tw * UP_RES_RATIO, self.TAG_HEIGHT * UP_RES_RATIO);
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

ImageLabel.prototype.addShape = function() {
};

ImageLabel.prototype.splitShape = function() {
};

/**
 * Returns whether or not this label allows switching to another SatItem
 * @return {boolean}
 */
ImageLabel.prototype.allowsLeavingCurrentItem = function() {
  return true;
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

ImageLabel.prototype.keyup = function(e) { // eslint-disable-line

};
