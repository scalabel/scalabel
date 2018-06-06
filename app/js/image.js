/* global SatItem SatLabel */

/* exported SatImage ImageLabel */

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
  self.hoverLabel = null;

  self.MAX_SCALE = 3.0;
  self.MIN_SCALE = 1.0;
  self.SCALE_RATIO = 1.5;

  self._isMouseDown = false;
}

SatImage.prototype = Object.create(SatItem.prototype);

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
    self.imageCanvas = document.getElementById('image_canvas');
    self.hiddenCanvas = document.getElementById('hidden_canvas');
    self.mainCtx = self.imageCanvas.getContext('2d');
    self.hiddenCtx = self.hiddenCanvas.getContext('2d');
    self.state = 'free';
    self.lastLabelID = -1;
    self.padBox = self._getPadding();
    self.catSel = document.getElementById('category_select');
    self.catSel.selectedIndex = 0;
    self.occlCheckbox = document.getElementById('occluded_checkbox');
    self.truncCheckbox = document.getElementById('truncated_checkbox');
    document.getElementById('prev_btn').onclick = function() {
      self.sat.gotoItem(self.index - 1);
    };
    document.getElementById('next_btn').onclick = function() {
      self.sat.gotoItem(self.index + 1);
    };

    // there may be some tension between this and the above block of code
    // TODO: test
    self.imageCanvas.height = self.imageHeight;
    self.imageCanvas.width = self.imageWidth;
    self.hiddenCanvas.height = self.imageHeight;
    self.hiddenCanvas.width = self.imageWidth;
    self.setScale(self.MIN_SCALE);

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
    document.onscroll = function(e) {
      self._scroll(e);
    };

    // buttons
    document.getElementById('prev_btn').onclick = function() {
      self.sat.gotoItem(self.index - 1);
    };
    document.getElementById('next_btn').onclick = function() {
      self.sat.gotoItem(self.index + 1);
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
          self.redraw();
        }
      });
    }
    if (deleteBtn.length) {
      deleteBtn.click(function() {
        if (self.selectedLabel) {
          self.selectedLabel.delete();
          self.redraw();
        }
      });
    }
    if (removeBtn.length) {
      removeBtn.click(function() {
        if (self.selectedLabel) {
          self.selectedLabel.delete();
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

    self.lastLabelID = 0;
    self.padBox = self._getPadding();
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
};

/**
 * Redraws this SatImage and all labels.
 */
SatImage.prototype.redraw = function() {
  let self = this;
  self.deleteInvalidLabels();
  self.padBox = self._getPadding();
  self.mainCtx.clearRect(0, 0, self.imageCanvas.width,
    self.imageCanvas.height);
  self.hiddenCtx.clearRect(0, 0, self.hiddenCanvas.width,
    self.hiddenCanvas.height);
  self.mainCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
    self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
  for (let i = 0; i < self.labels.length; i++) {
    self.labels[i].redraw(self.mainCtx, self.hiddenCtx, self.selectedLabel,
      self.hoverLabel, i);
  }
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
 * Called when this SatImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousedown = function(e) {
  let self = this;
  let mousePos = self._getMousePos(e);
  if (self._isWithinFrame(e) && self.state === 'free') {
    [self.selectedLabel, self.currHandle] = self._getSelected(mousePos);
    // change checked traits on label selection
    if (self.selectedLabel) {
      self.selectedLabel.currHandle = self.currHandle;
      for (let i = 0; i < self.catSel.options.length; i++) {
        if (self.catSel.options[i].innerHTML ===
          self.selectedLabel.categoryPath) {
          self.catSel.selectedIndex = i;
          break;
        }
      }
      if ($('[name=\'occluded-checkbox\']').prop('checked') !==
        self.selectedLabel.occl) {
        $('[name=\'occluded-checkbox\']').trigger('click');
      }
      if ($('[name=\'truncated-checkbox\']').prop('checked') !==
        self.selectedLabel.trunc) {
        $('[name=\'truncated-checkbox\']').trigger('click');
      }
      // TODO: Wenqi
      // traffic light color
    }

    if (self.selectedLabel && self.currHandle > 0) {
      // if we have a resize handle
      self.state = 'resize';
      self.selectedLabel.state = 'resize';
      self.resizeID = self.selectedLabel.id;
    } else if (self.currHandle === 0 && self.selectedLabel) {
      // if we have a move handle
      self.selectedLabel.movePos = self.selectedLabel.getCurrentPosition();
      self.selectedLabel.moveClickPos = mousePos;
      self.state = 'move';
      self.selectedLabel.state = 'move';
    } else if (!self.selectedLabel) {
      // otherwise, new label
      let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
      let occl = self.occlCheckbox.checked;
      let trunc = self.truncCheckbox.checked;
      self.selectedLabel = self.sat.newLabel({
        categoryPath: cat, occl: occl,
        trunc: trunc, mousePos: mousePos,
      });
      self.selectedLabel.state = 'resize';
      self.state = 'resize';
      self.currHandle = self.selectedLabel.INITIAL_HANDLE;
      self.resizeID = self.selectedLabel.id;
    }
  }
  if (!this._isWithinFrame(e)) {
    return;
  }
  self._isMouseDown = true;
  this.redraw();
};

/**
 * Called when this SatImage is active and the mouse is moved.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousemove = function(e) {
  let mousePos = this._getMousePos(e);
  if (this.sat.LabelType.useCrossHair) {
    this.drawCrossHair(e);
  }
  if (this._isWithinFrame(e)) {
    this.imageCanvas.style.cursor = 'auto';
    // hover effect
    let hoverHandle;
    [this.hoverLabel, hoverHandle] = this._getSelected(mousePos);

    if (this.hoverLabel) {
      this.hoverLabel.setCurrHoverHandle(hoverHandle);
    }
    // label specific handling of mousemove
    if (this.selectedLabel) {
      this.selectedLabel.mousemove(e);
    }

    if (this._isMouseDown && this.selectedLabel) {
      this.imageCanvas.style.cursor = this.selectedLabel.getCursorStyle(
        this.selectedLabel.getCurrHandle());
    } else if (!this._isMouseDown && this.hoverLabel) {
      this.imageCanvas.style.cursor = this.hoverLabel.getCursorStyle(
        this.hoverLabel.getCurrHoverHandle());
    } else {
      this.imageCanvas.style.cursor = 'crosshair';
    }
  }
  this.redraw();
};

/**
 * Called when this SatImage is active and the mouse is released.
 * @param {object} _: mouse event (unused)
 */
SatImage.prototype._mouseup = function(_) { // eslint-disable-line
  this._isMouseDown = false;
  this.state = 'free';

  if (this.selectedLabel) {
    // label specific handling of mouseup
    this.selectedLabel.mouseup();
    if (this.selectedLabel.isSmall()) {
      this.selectedLabel.delete();
    }
  }
  this.redraw();
};

/**
 * Called when this SatImage is active and the mouse is scrolled.
 * @param {object} e: mouse event
 */
SatImage.prototype._scroll = function(e) {
  let self = this;
  if (self.sat.LabelType.useCrossHair) {
    self.drawCrossHair(e);
  }
  self.redraw();
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
SatImage.prototype._getMousePos = function(e) {
  let self = this;
  let rect = self.imageCanvas.getBoundingClientRect();
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
    this.occl = $('[name=\'occluded-checkbox\']').prop('checked');
  }
};

/**
 * Called when the truncated checkbox is toggled.
 */
SatImage.prototype._truncSwitch = function() {
  if (this.selectedLabel) {
    this.trunc = $('[name=\'truncated-checkbox\']').prop(
      'checked');
  }
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
 * Called when the traffic light color choice is changed.
 */
SatImage.prototype._lightSwitch = function() {
  // TODO: Wenqi
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
 * @param {number} id: label object identifier
 * @param {object} optionalAttributes: Optional attributes for the SatLabel.
 */
function ImageLabel(sat, id, optionalAttributes = null) {
  SatLabel.call(this, sat, id, optionalAttributes);
  this.image = sat.currentItem;
}

ImageLabel.prototype = Object.create(SatLabel.prototype);

ImageLabel.useCrossHair = false;

ImageLabel.prototype.getCurrentPosition = function() {

};

ImageLabel.prototype.fromJsonPointers = function(json) {
  let self = this;
  self.decodeBaseJsonPointers(json);
  self.image = self.sat.currentItem;
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

/**
 * Draw a specified resize handle of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
ImageLabel.prototype.drawHandle = function(ctx, handleNo) {
  let self = this;
  ctx.save(); // save the canvas context settings
  let posHandle = self.getHandle(handleNo);

  [posHandle.x, posHandle.y] = self.image.transformPoints(
    [posHandle.x, posHandle.y]);

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
