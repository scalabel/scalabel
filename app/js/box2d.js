/* global SatImage ImageLabel */
/* exported Box2d Box2dImage*/

/**
 * 2D box image labeling
 * @param {Sat} sat: task context
 * @param {int} index: index of this image in the task
 * @param {string} url: source of the image
 * @constructor
 */
function Box2dImage(sat, index, url) {
  SatImage.call(this, sat, index, url);

  this.active = false; // whether this is the currently displayed image
}

Box2dImage.prototype = Object.create(SatImage.prototype);

/**
 * Make this Box2dImage the active one in the sat instance. This is effectively
 *  the true initialization of Box2dImage, as it connects this to all of the
 *  page elements.
 * TODO: do we need makeInactive?
 */
Box2dImage.prototype.makeActive = function() {
  let self = this;
  self.active = true;
  self.imageCanvas = document.getElementById('image_canvas');
  self.hiddenCanvas = document.getElementById('hidden_canvas');
  self.mainCtx = self.imageCanvas.getContext('2d');
  self.hiddenCtx = self.hiddenCanvas.getContext('2d');
  self.state = 'free';
  self.lastLabelID = 0;
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
  document.onmousedown = function(e) {
    self._mousedown(e);
  };
  document.onmousemove = function(e) {
    self._mousemove(e);
  };
  document.onmouseup = function(e) {
    self._mouseup(e);
  };
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
  $('#remove_btn').click(function() {
    self.remove();
  });
};

/**
 * Called when this Box2dImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
Box2dImage.prototype._mousedown = function(e) {
  let self = this;
  if (self._isWithinFrame(e) && self.state === 'free') {
    let mousePos = self._getMousePos(e);
    [self.selBox, self.currHandle] = self._getSelected(mousePos);

    // change checked traits on box selection
    if (self.selBox) {
      if ($('[name=\'occluded-checkbox\']').prop('checked') !==
        self.selBox.occl) {
        $('[name=\'occluded-checkbox\']').trigger('click');
      }
      if ($('[name=\'truncated-checkbox\']').prop('checked') !==
        self.selBox.trunc) {
        $('[name=\'truncated-checkbox\']').trigger('click');
      }
      // TODO: Wenqi
      // traffic light color
    }

    if (self.selBox && self.currHandle >= 0 && self.currHandle <= 7) {
      // if we have a resize handle
      self.state = 'resize';
      self.resizeID = self.selBox.id;
    } else if (self.currHandle === 8) {
      // if we have a move handle
      self.movePos = {x: self.selBox.x, y: self.selBox.y, w: self.selBox.w,
        h: self.selBox.h};
      self.moveClickPos = mousePos;
      self.state = 'move';
    } else if (!self.selBox) {
      // otherwise, new box
      let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
      let occl = self.occlCheckbox.checked;
      let trunc = self.truncCheckbox.checked;
      self.lastLabelID += 1;
      self.selBox = new Box2d(self.sat, self.lastLabelID, cat, occl, trunc,
        mousePos);
      self.labels.push(self.selBox);
      self.state = 'resize';
      // treat it like resize handle of 3 (bottom-right)
      self.currHandle = 3;
      self.resizeID = self.selBox.id;
    }
  }
  self.redraw();
};

/**
 * Called when this Box2dImage is active and the mouse is moved.
 * @param {object} e: mouse event
 */
Box2dImage.prototype._mousemove = function(e) {
  let self = this;
  let canvRect = this.imageCanvas.getBoundingClientRect();
  let mousePos = self._getMousePos(e);

  // draw the crosshair
  let cH = $('#crosshair-h');
  let cV = $('#crosshair-v');
  cH.css('top', Math.min(canvRect.y + self.padBox.y + self.padBox.h, Math.max(
    e.clientY, canvRect.y + self.padBox.y)));
  cH.css('left', canvRect.x + self.padBox.x);
  cH.css('width', self.padBox.w);
  cV.css('left', Math.min(canvRect.x + self.padBox.x + self.padBox.w, Math.max(
    e.clientX, canvRect.x + self.padBox.x)));
  cV.css('top', canvRect.y + self.padBox.y);
  cV.css('height', self.padBox.h);
  if (self._isWithinFrame(e)) {
    $('.hair').show();
  } else {
    $('.hair').hide();
  }

  // needed for on-hover animations
  [self.hoverBox, self.hoverHandle] = self._getSelected(mousePos);
  // change the cursor appropriately
  if (self.state === 'resize') {
    self.imageCanvas.style.cursor = 'crosshair';
  } else if (self.state === 'move') {
    self.imageCanvas.style.cursor = 'move';
  } else {
    switch (self.hoverHandle) {
      case 0:
        self.imageCanvas.style.cursor = 'nwse-resize';
        break;
      case 1:
        self.imageCanvas.style.cursor = 'nesw-resize';
        break;
      case 2:
        self.imageCanvas.style.cursor = 'nesw-resize';
        break;
      case 3:
        self.imageCanvas.style.cursor = 'nwse-resize';
        break;
      case 4:
        self.imageCanvas.style.cursor = 'ns-resize';
        break;
      case 5:
        self.imageCanvas.style.cursor = 'ew-resize';
        break;
      case 6:
        self.imageCanvas.style.cursor = 'ns-resize';
        break;
      case 7:
        self.imageCanvas.style.cursor = 'ew-resize';
        break;
      case 8:
        self.imageCanvas.style.cursor = 'move';
        break;
      case null:
        self.imageCanvas.style.cursor = 'crosshair';
        break;
    }
  }

  if (self.state === 'resize') {
    if ([0, 4, 1].indexOf(self.currHandle) > -1) {
      let newY = Math.min(canvRect.height - self.padBox.y, Math.max(
        self.padBox.y, mousePos.y));
      self.selBox.h += self.selBox.y - newY;
      self.selBox.y = newY;
    }
    if ([1, 7, 3].indexOf(self.currHandle) > -1) {
      self.selBox.w = Math.min(canvRect.width - self.padBox.x - self.selBox.x,
        Math.max(self.padBox.x - self.selBox.x, mousePos.x - self.selBox.x));
    }
    if ([3, 6, 2].indexOf(self.currHandle) > -1) {
      self.selBox.h = Math.min(canvRect.height - self.padBox.y - self.selBox.y,
        Math.max(self.padBox.y - self.selBox.y, mousePos.y - self.selBox.y));
    }
    if ([2, 5, 0].indexOf(self.currHandle) > -1) {
      let newX = Math.min(canvRect.width - self.padBox.x, Math.max(
        self.padBox.x, mousePos.x));
      self.selBox.w += self.selBox.x - newX;
      self.selBox.x = newX;
    }
  } else if (self.state === 'move') {
    // get the delta and correct to account for max distance
    let delta = {x: mousePos.x - self.moveClickPos.x, y: mousePos.y -
      self.moveClickPos.y};
    let minX = Math.min(self.movePos.x, self.movePos.x + self.movePos.w);
    let maxX = Math.max(self.movePos.x, self.movePos.x + self.movePos.w);
    let minY = Math.min(self.movePos.y, self.movePos.y + self.movePos.h);
    let maxY = Math.max(self.movePos.y, self.movePos.y + self.movePos.h);
    delta.x = Math.max(self.padBox.x - minX, delta.x);
    delta.x = Math.min(self.padBox.x + self.padBox.w - maxX, delta.x);
    delta.y = Math.max(self.padBox.y - minY, delta.y);
    delta.y = Math.min(self.padBox.y + self.padBox.h - maxY, delta.y);
    // update
    self.selBox.x = self.movePos.x + delta.x;
    self.selBox.y = self.movePos.y + delta.y;
  }
  self.redraw();
};

/**
 * Called when this Box2dImage is active and the mouse is released.
 * @param {object} _: mouse event (unused)
 */
Box2dImage.prototype._mouseup = function(_) { // eslint-disable-line
  let self = this;
  if (self.state !== 'free') {
    if (self.state === 'resize') {
      // if we resized, we need to reorder ourselves
      if (self.selBox.w < 0) {
        self.selBox.x = self.selBox.x + self.selBox.w;
        self.selBox.w = -1 * self.selBox.w;
      }
      if (self.selBox.h < 0) {
        self.selBox.y = self.selBox.y + self.selBox.h;
        self.selBox.h = -1 * self.selBox.h;
      }
      // remove the box if it's too small
      if (self.selBox.isSmall()) {
        self.remove();
      }
    }
    self.state = 'free';
    self.resizeID = null;
    self.movePos = null;
    self.moveClickPos = null;
  }
  self.redraw();
};

/**
 * Called when the selected category is changed.
 */
Box2dImage.prototype._changeCat = function() {
  let self = this;
  if (self.selBox) {
    let selOpt = self.catSel.options[self.catSel.selectedIndex].innerHTML;
    self.selBox.category = selOpt;
  }
};

/**
 * Called when the occluded checkbox is toggled.
 */
Box2dImage.prototype._occlSwitch = function() {
  let self = this;
  if (self.selBox) {
    self.selBox.occl = $('[name=\'occluded-checkbox\']').prop('checked');
  }
};

/**
 * Called when the truncated checkbox is toggled.
 */
Box2dImage.prototype._truncSwitch = function() {
  let self = this;
  if (self.selBox) {
    self.selBox.trunc = $('[name=\'truncated-checkbox\']').prop('checked');
  }
};

/**
 * Called when the traffic light color choice is changed.
 */
Box2dImage.prototype._lightSwitch = function() {
  // TODO: Wenqi
};

/**
 * Removes the currently selected box.
 */
Box2dImage.prototype.remove = function() {
  let self = this;
  if (self.selBox) {
    for (let i = 0; i < self.labels.length; i++) {
      if (self.labels[i].id === self.selBox.id) {
        self.labels.splice(i, 1);
        self.selBox = null;
        return;
      }
    }
  }
};

/**
 * Draws all of the boxes.
 */
Box2dImage.prototype.redraw = function() {
  let self = this;
  self.padBox = self._getPadding();
  self.mainCtx.clearRect(0, 0, self.imageCanvas.width,
    self.imageCanvas.height);
  self.hiddenCtx.clearRect(0, 0, self.hiddenCanvas.width,
    self.hiddenCanvas.height);
  self.mainCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
    self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
  for (let i = 0; i < self.labels.length; i++) {
    self.labels[i].redraw(self.mainCtx, self.hiddenCtx, self.selBox,
      self.resizeID === i, self.hoverBox, self.hoverHandle);
  }
};

/**
 * True if mouse is within the image frame (tighter bound than canvas).
 * @param {object} e: mouse event
 * @return {boolean}: whether the mouse is within the image frame
 */
Box2dImage.prototype._isWithinFrame = function(e) {
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
Box2dImage.prototype._getMousePos = function(e) {
  let rect = this.imageCanvas.getBoundingClientRect();
  return {x: e.clientX - rect.x, y: e.clientY - rect.y};
};

/**
 * Get the padding for the image given its size and canvas size.
 * @return {object}: padding box (x,y,w,h)
 */
Box2dImage.prototype._getPadding = function() {
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
 * Get the box with a given id.
 * @param {number} boxID: id of the sought box
 * @return {Box2d}: the sought box
 */
Box2dImage.prototype._getBoxByID = function(boxID) {
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].id === boxID) {
      return this.labels[i];
    }
  }
};

/**
 * Get the box and handle under the mouse.
 * @param {object} mousePos: canvas mouse position (x,y)
 * @return {[Box2d, number]}: the box and handle (0-9) under the mouse
 */
Box2dImage.prototype._getSelected = function(mousePos) {
  let pixelData = this.hiddenCtx.getImageData(mousePos.x,
    mousePos.y, 1, 1).data;
  let selBoxID = null;
  let currHandle = null;
  if (pixelData[0] !== 0 && pixelData[3] === 255) {
    selBoxID = pixelData[0] - 1;
    currHandle = pixelData[1] - 1;
  }
  return [this._getBoxByID(selBoxID), currHandle];
};

/**
 * 2D box label
 * @param {Sat} sat: context
 * @param {int} id: label id
 * @param {string} category: label category
 * @param {boolean} occl: Whether box is occluded
 * @param {boolean} trunc: Whether box is truncated
 * @param {object} mousePos: mouse position (x,y) on canvas
 */
function Box2d(sat, id, category, occl, trunc, mousePos) {
  ImageLabel.call(this, sat, id);

  this.category = category;
  this.occl = occl;
  this.trunc = trunc;

  this.x = mousePos.x;
  this.y = mousePos.y;
  this.w = 0;
  this.h = 0;

  // constants
  this.LINE_WIDTH = 2;
  this.OUTLINE_WIDTH = 1;
  this.HANDLE_RADIUS = 4;
  this.HIDDEN_LINE_WIDTH = 4;
  this.HIDDEN_HANDLE_RADIUS = 7;
  this.TAG_WIDTH = 25;
  this.TAG_HEIGHT = 14;
  this.MIN_BOX_SIZE = 15;
  this.FADED_ALPHA = 0.5;
}

Box2d.prototype = Object.create(ImageLabel.prototype);

/**
 * Draw this bounding box on the canvas.
 * @param {object} mainCtx - HTML canvas context for visible objects.
 * @param {object} hiddenCtx - HTML canvas context for hidden objects.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param {boolean} resizing - Whether or not this box is being resized.
 * @param{number} hoverBox - ID of the currently hovered over box, or null if
 *   no box hovered over.
 * @param {number} hoverHandle - handle number of the currently hovered handle,
 *   or null if no handle hovered.
 */
Box2d.prototype.redraw = function(mainCtx, hiddenCtx, selectedBox, resizing,
  hoverBox, hoverHandle) {
  let self = this;

  // go ahead and set context font
  mainCtx.font = '11px Verdana';

  // draw visible elements
  self.drawBox(mainCtx, selectedBox, resizing);
  self.drawHandles(mainCtx, selectedBox, hoverBox, hoverHandle);
  self.drawTag(mainCtx);

  // draw hidden elements
  self.drawHiddenBox(hiddenCtx, selectedBox);
  self.drawHiddenHandles(hiddenCtx, selectedBox);
};

/**
 * Draw the box part of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param {boolean} resizing - Whether or not this box is being resized.
 */
Box2d.prototype.drawBox = function(ctx, selectedBox, resizing) {
  let self = this;
  ctx.save(); // save the canvas context settings
  if (selectedBox && selectedBox.id != self.id) {
    // if exists selected box and it's not this one, alpha this out
    ctx.globalAlpha = self.FADED_ALPHA;
  }
  if (resizing) {
    ctx.setLineDash([3]); // if box is being resized, use line dashes
  }
  if (self.isSmall()) {
    ctx.strokeStyle = 'rgb(169, 169, 169)'; // if box is too small, gray it out
  } else {
    // otherwise use regular color
    ctx.strokeStyle = self.styleColor();
  }
  ctx.lineWidth = self.LINE_WIDTH; // set line width
  ctx.strokeRect(self.x, self.y, self.w, self.h); // draw the box
  ctx.restore(); // restore the canvas to saved settings
};

/**
 * Draw the resize handles of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param{number} hoverBox - ID of the currently hovered over box, or null if
 *   no box hovered over.
 * @param {number} hoverHandle - handle number of the currently hovered handle,
 *   or null if no handle hovered.
 */
Box2d.prototype.drawHandles = function(ctx, selectedBox, hoverBox,
  hoverHandle) {
  let self = this;
  if (selectedBox && selectedBox.id === self.id) {
    // if this box is selected, draw all its handles
    for (let handleNo = 0; handleNo < 8; handleNo++) {
      self.drawHandle(ctx, handleNo);
    }
  } else if (!selectedBox && hoverBox && hoverBox.id === self.id
    && hoverHandle) {
    // else if no selection and a handle is hovered over, draw it
    self.drawHandle(ctx, hoverHandle);
  }
};

/**
 * Draw a specified resize handle of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
Box2d.prototype.drawHandle = function(ctx, handleNo) {
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

/**
 * Draw the label tag of this bounding box.
 * @param {object} ctx - Canvas context.
 */
Box2d.prototype.drawTag = function(ctx) {
  let self = this;
  if (!self.isSmall()) {
    ctx.save();
    let words = self.category.split(' ');
    let tw = self.TAG_WIDTH;
    // abbreviate tag as the first 3 chars of the last word
    let abbr = words[words.length - 1].substring(0, 3);
    if (self.occl) {
      abbr += ',o';
      tw += 9;
    }
    if (self.trunc) {
      abbr += ',t';
      tw += 9;
    }
    // get the top left corner
    let tlx = Math.min(self.x, self.x + self.w);
    let tly = Math.min(self.y, self.y + self.h);
    ctx.fillStyle = self.styleColor();
    ctx.fillRect(tlx + 1, tly - self.TAG_HEIGHT, tw,
      self.TAG_HEIGHT);
    ctx.fillStyle = 'rgb(0,0,0)';
    ctx.fillText(abbr, tlx + 3, tly - 3);
    ctx.restore();
  }
};

/**
 * Draw the box part of the hidden box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 */
Box2d.prototype.drawHiddenBox = function(hiddenCtx, selectedBox) {
  // only draw if it is not the case that there is another selected box
  let self = this;
  if (!selectedBox || selectedBox.id === self.id) {
    hiddenCtx.save(); // save the canvas context settings
    // 8 represents the box itself
    hiddenCtx.strokeStyle = self.hiddenStyleColor(8);
    hiddenCtx.lineWidth = self.HIDDEN_LINE_WIDTH;
    hiddenCtx.strokeRect(self.x, self.y, self.w, self.h); // draw the box
    hiddenCtx.restore(); // restore the canvas to saved settings
  }
};

/**
 * Draw the hidden resize handles of this bounding box.
 * @param {object} hiddenCtx - Hidden canvas context
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected
 */
Box2d.prototype.drawHiddenHandles = function(hiddenCtx, selectedBox) {
  let self = this;
  if (!selectedBox || selectedBox.id === self.id) {
    // as long as there is not another box selected, draw all the hidden handles
    for (let handleNo = 0; handleNo < 8; handleNo++) {
      self.drawHiddenHandle(hiddenCtx, handleNo);
    }
  }
};

/**
 * Draw a specified hidden resize handle of this bounding box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
Box2d.prototype.drawHiddenHandle = function(hiddenCtx, handleNo) {
  let self = this;
  hiddenCtx.save(); // save the canvas context settings
  let posHandle = self._getHandle(handleNo);
  hiddenCtx.fillStyle = self.hiddenStyleColor(handleNo);
  hiddenCtx.lineWidth = self.HIDDEN_LINE_WIDTH;
  hiddenCtx.beginPath();
  hiddenCtx.arc(posHandle.x, posHandle.y, self.HIDDEN_HANDLE_RADIUS, 0,
    2 * Math.PI);
  hiddenCtx.fill();
  hiddenCtx.restore(); // restore the canvas to saved settings
};

/**
 * Get whether this bounding box is too small.
 * @return {boolean} - True if the box is too small.
 */
Box2d.prototype.isSmall = function() {
  return Math.min(Math.abs(this.w), Math.abs(this.h)) < this.MIN_BOX_SIZE;
};

/**
 * Get the hidden color as rgb, which encodes the id and handle index.
 * @param {number} handleNo - The handle number, ranges from 0 to 8.
 * @return {string} - The hidden color rgb string.
 */
Box2d.prototype.hiddenStyleColor = function(handleNo) {
  return ['rgb(' + (this.id + 1), handleNo + 1, '0)'].join(',');
};

/**
 * Converts handle number to the central point of specified resize handle.
 * @param {number} handleNo - The handle number, ranges from 0 to 8.
 * @return {object} - A struct with x and y of the handle's center.
 */
Box2d.prototype._getHandle = function(handleNo) {
  let self = this;
  switch (handleNo) {
    case 0:
      return {x: self.x, y: self.y};
    case 1:
      return {x: self.x + self.w, y: self.y};
    case 2:
      return {x: self.x, y: self.y + self.h};
    case 3:
      return {x: self.x + self.w, y: self.y + self.h};
    case 4:
      return {x: self.x + self.w / 2, y: self.y};
    case 5:
      return {x: self.x, y: self.y + self.h / 2};
    case 6:
      return {x: self.x + self.w / 2, y: self.y + self.h};
    case 7:
      return {x: self.x + self.w, y: self.y + self.h / 2};
    case 8:
      return; // this shouldn't happen (8 is for the box)
  }
};
