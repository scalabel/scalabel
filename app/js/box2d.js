/* global ImageLabel */
/* exported Box2d*/

/**
 * 2D box label
 * @param {Sat} sat: context
 * @param {int} id: label id
 * @param {object} kargs: arguments of the box, containing category,
    occl/trunc, and mousePos
 */
function Box2d(sat, id, kargs) {
  // TODO: separate category and attributes in kargs
  ImageLabel.call(this, sat, id);

  this.name = kargs.category;
  // TODO: Move those to attributes
  // TODO: don't use abbreviation here
  this.occl = kargs.occl;
  this.trunc = kargs.trunc;

  // TODO: move the coordinates to one object
  this.x = kargs.mousePos.x;
  this.y = kargs.mousePos.y;
  this.w = 0;
  this.h = 0;

  // constants
  // TODO: Move out, we don't have to keep a copy of constants in every object
  this.LINE_WIDTH = 2;
  this.OUTLINE_WIDTH = 1;
  this.HANDLE_RADIUS = 4;
  this.HIDDEN_LINE_WIDTH = 4;
  this.HIDDEN_HANDLE_RADIUS = 7;
  this.TAG_WIDTH = 25;
  this.TAG_HEIGHT = 14;
  this.MIN_BOX_SIZE = 15;
  this.FADED_ALPHA = 0.5;
  this.INITIAL_HANDLE = 5; // for the bottom-right of the box
}

Box2d.prototype = Object.create(ImageLabel.prototype);

Box2d.prototype.toJson = function() {
  let self = this;
  let json = ImageLabel.prototype.toJson();
  json.box2d = {x: self.x, y: self.y, w: self.w, h: self.h};
  // TODO: remove this special attribute assignment
  json.attributes = {occlusion: self.occl, truncation: self.trunc};
  return json;
};

/**
 * Load label information from json object
 * @param {object} json: JSON representation of this Box2d.
 */
Box2d.prototype.fromJson = function(json) {
  let self = this;
  ImageLabel.prototype.fromJson(json);
  self.x = json.box2d.x;
  self.y = json.box2d.y;
  self.w = json.box2d.w;
  self.h = json.box2d.h;
  // TODO: stop hardcoding occl and trunc attributes
  self.occl = json.attributes.occlusion;
  self.trunc = json.attributes.truncation;
};

Box2d.prototype.toJson = function() {
  let self = this;
  let json = ImageLabel.prototype.toJson();
  json.box2d = {x: self.x, y: self.y, w: self.w, h: self.h};
  // TODO: remove this special attribute assignment
  json.attributes = {occlusion: self.occl, truncation: self.trunc};
  return json;
};

/**
 * Load label information from json object
 * @param {object} json: JSON representation of this Box2d.
 */
Box2d.prototype.fromJson = function(json) {
  let self = this;
  ImageLabel.prototype.fromJson(json);
  self.x = json.box2d.x;
  self.y = json.box2d.y;
  self.w = json.box2d.w;
  self.h = json.box2d.h;
  // TODO: stop hardcoding occl and trunc attributes
  self.occl = json.attributes.occlusion;
  self.trunc = json.attributes.truncation;
};

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
 * @param {number} labelIndex - index of this label in this.sat.labels
 */
Box2d.prototype.redraw = function(mainCtx, hiddenCtx, selectedBox, resizing,
  hoverBox, hoverHandle, labelIndex) {
  let self = this;

  // go ahead and set context font
  mainCtx.font = '11px Verdana';

  // draw visible elements
  self.drawBox(mainCtx, selectedBox, resizing);
  self.drawTag(mainCtx);
  if (selectedBox && self.id === selectedBox.id) {
    self.drawHandles(mainCtx);
  }

  // draw hidden elements
  self.drawHiddenBox(hiddenCtx, selectedBox, labelIndex);
  self.drawHiddenHandles(hiddenCtx, selectedBox, labelIndex);
};

/**
 * Draw the box part of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {object} selectedBox - The currently selected box, or null if no
 *   box selected.
 * @param {number} selectedBox.id - ID of the currently selected box.
 * @param {boolean} resizing - Whether or not this box is being resized.
 */
Box2d.prototype.drawBox = function(ctx, selectedBox, resizing) {
  let self = this;
  ctx.save(); // save the canvas context settings
  if (selectedBox && selectedBox.id !== self.id) {
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
    for (let handleNo = 1; handleNo <= 8; handleNo++) {
      self.drawHandle(ctx, handleNo);
    }
  } else if (!selectedBox && hoverBox && hoverBox.id === self.id
    && hoverHandle > 0) {
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
    let words = self.name.split(' ');
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
 * @param {number} labelIndex - index of this label in this.sat.labels
 */
Box2d.prototype.drawHiddenBox = function(hiddenCtx, selectedBox, labelIndex) {
  // only draw if it is not the case that there is another selected box
  let self = this;
  if (!selectedBox || selectedBox.id === self.id) {
    hiddenCtx.save(); // save the canvas context settings
    // 0 represents the box itself
    hiddenCtx.strokeStyle = self.hiddenStyleColor(labelIndex, 0);
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
 * @param {number} labelIndex - index of this label in this.sat.labels
 */
Box2d.prototype.drawHiddenHandles = function(hiddenCtx, selectedBox,
  labelIndex) {
  let self = this;
  if (!selectedBox || selectedBox.id === self.id) {
    // as long as there is not another box selected, draw all the hidden handles
    for (let handleNo = 1; handleNo <= 8; handleNo++) {
      self.drawHiddenHandle(hiddenCtx, handleNo, labelIndex);
    }
  }
};

/**
 * Draw a specified hidden resize handle of this bounding box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 * @param {number} labelIndex - index of this label in this.sat.labels
 */
Box2d.prototype.drawHiddenHandle = function(hiddenCtx, handleNo, labelIndex) {
  let self = this;
  hiddenCtx.save(); // save the canvas context settings
  let posHandle = self._getHandle(handleNo);
  hiddenCtx.fillStyle = self.hiddenStyleColor(labelIndex, handleNo);
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
 * @param {number} labelIndex - index of this label in this.sat.labels
 * @param {number} handleNo - The handle number, ranges from 0 to 8.
 * @return {string} - The hidden color rgb string.
 */
Box2d.prototype.hiddenStyleColor = function(labelIndex, handleNo) {
  return ['rgb(' + (Math.floor(labelIndex / 256)), labelIndex % 256,
    (handleNo + 1) + ')'].join(',');
};

/**
 * Get the cursor style for a specified handle number.
 * @param {int} handleNo - The handle number, ranges from 0 to 8.
 * @return {string} - The cursor style string.
 */
Box2d.prototype.getCursorStyle = function(handleNo) {
  return ['move', 'nwse-resize', 'ns-resize', 'nesw-resize', 'ew-resize',
    'nwse-resize', 'ns-resize', 'nesw-resize', 'ew-resize'][handleNo];
};

/**
 * Resizes this box based on the current mouse position.
 * @param {object} mousePos - The x, y mouse position on the canvas.
 * @param {number} currHandle - The numerical index of the current handle.
 * @param {object} canvRect - The box of the canvas position.
 * @param {object} padBox - The padded box inside the canvas.
 */
Box2d.prototype.resize = function(mousePos, currHandle, canvRect, padBox) {
  let self = this;
  if ([1, 2, 3].indexOf(currHandle) > -1) {
    let newY = Math.min(canvRect.height - padBox.y, Math.max(
      padBox.y, mousePos.y));
    self.h += self.y - newY;
    self.y = newY;
  }
  if ([3, 4, 5].indexOf(currHandle) > -1) {
    self.w = Math.min(canvRect.width - padBox.x - self.x,
      Math.max(padBox.x - self.x, mousePos.x - self.x));
  }
  if ([5, 6, 7].indexOf(currHandle) > -1) {
    self.h = Math.min(canvRect.height - padBox.y - self.y,
      Math.max(padBox.y - self.y, mousePos.y - self.y));
  }
  if ([7, 8, 1].indexOf(currHandle) > -1) {
    let newX = Math.min(canvRect.width - padBox.x, Math.max(
      padBox.x, mousePos.x));
    self.w += self.x - newX;
    self.x = newX;
  }
  if (self.parent) {
    self.sat.interpolate(self); // TODO
  }
};

/**
 * Moves this box based on the current mouse position.
 * @param {object} mousePos - The x, y mouse position on the canvas.
 * @param {object} movePos - The box position of the original box.
 * @param {object} moveClickPos - The x, y position of the original click.
 * @param {object} padBox - The padded box inside the canvas.
 */
Box2d.prototype.move = function(mousePos, movePos, moveClickPos, padBox) {
  let self = this;
  // get the delta and correct to account for max distance
  let delta = {x: mousePos.x - moveClickPos.x, y: mousePos.y - moveClickPos.y};
  let minX = Math.min(movePos.x, movePos.x + movePos.w);
  let maxX = Math.max(movePos.x, movePos.x + movePos.w);
  let minY = Math.min(movePos.y, movePos.y + movePos.h);
  let maxY = Math.max(movePos.y, movePos.y + movePos.h);
  delta.x = Math.max(padBox.x - minX, delta.x);
  delta.x = Math.min(padBox.x + padBox.w - maxX, delta.x);
  delta.y = Math.max(padBox.y - minY, delta.y);
  delta.y = Math.min(padBox.y + padBox.h - maxY, delta.y);
  // update
  self.x = movePos.x + delta.x;
  self.y = movePos.y + delta.y;
  if (self.parent) {
    self.sat.interpolate(self); // TODO
  }
};

/**
 * Get the current position of this box.
 * @return {object} - The box's current position.
 */
Box2d.prototype.getCurrentPosition = function() {
    let self = this;
    return {x: self.x, y: self.y, w: self.w, h: self.h};
};

/**
 * Get the weighted average between this box and a provided box.
 * @param {ImageLabel} box - The other box.
 * @param {number} weight - The weight, b/w 0 and 1, higher corresponds to
 *   closer to the other box.
 * @return {object} - The box's position.
 */
Box2d.prototype.getWeightedAvg = function(box, weight) {
  let self = this;
  let avg = {};
  avg.x = self.x + weight*(box.x - self.x);
  avg.y = self.y + weight*(box.y - self.y);
  avg.w = self.w + weight*(box.w - self.w);
  avg.h = self.h + weight*(box.h - self.h);
  return avg;
};

/**
 * Set this box to be the weighted average of the two provided boxes.
 * @param {Box2d} startBox - The first box.
 * @param {Box2d} endBox - The second box.
 * @param {number} weight - The weight, b/w 0 and 1, higher corresponds to
    closer to endBox.
 */
Box2d.prototype.weightedAvg = function(startBox, endBox, weight) {
  let self = this;
  let avg = startBox.getWeightedAvg(endBox, weight);
  self.x = avg.x;
  self.y = avg.y;
  self.w = avg.w;
  self.h = avg.h;
};

/**
 * Calculate the intersection between this and another Box2d.
 * @param {Box2d} box - The other box.
 * @return {number} - The intersection between the two boxes.
 */
Box2d.prototype.intersection = function(box) {
  let self = this;
  let b1x1 = Math.min(self.x, self.x + self.w);
  let b1x2 = Math.max(self.x, self.x + self.w);

  let b1y1 = Math.min(self.y, self.y + self.h);
  let b1y2 = Math.max(self.y, self.y + self.h);

  let b2x1 = Math.min(box.x, box.x + box.w);
  let b2x2 = Math.max(box.x, box.x + box.w);

  let b2y1 = Math.min(box.y, box.y + box.h);
  let b2y2 = Math.max(box.y, box.y + box.h);

  let ix1 = Math.max(b1x1, b2x1);
  let ix2 = Math.min(b1x2, b2x2);
  let iy1 = Math.max(b1y1, b2y1);
  let iy2 = Math.min(b1y2, b2y2);

  return Math.max(0, (ix2 - ix1) * (iy2 - iy1));
};

Box2d.prototype.union = function(box) {
  let self = this;
  let intersection = self.intersection(box);
  return Math.abs(self.w * self.h) + Math.abs(box.w * box.h) - intersection;
};

/**
 * Get the weighted average between this box and a provided box.
 * @param {ImageLabel} box - The other box.
 * @param {number} weight - The weight, b/w 0 and 1, higher corresponds to
 *   closer to the other box.
 * @return {object} - The box's position.
 */
Box2d.prototype.getWeightedAvg = function(box, weight) {
  let self = this;
  let avg = {};
  avg.x = self.x + weight*(box.x - self.x);
  avg.y = self.y + weight*(box.y - self.y);
  avg.w = self.w + weight*(box.w - self.w);
  avg.h = self.h + weight*(box.h - self.h);
  return avg;
};

/**
 * Set this box to be the weighted average of the two provided boxes.
 * @param {Box2d} startBox - The first box.
 * @param {Box2d} endBox - The second box.
 * @param {number} weight - The weight, b/w 0 and 1, higher corresponds to
 closer to endBox.
 */
Box2d.prototype.weightedAvg = function(startBox, endBox, weight) {
  let self = this;
  let avg = startBox.getWeightedAvg(endBox, weight);
  self.x = avg.x;
  self.y = avg.y;
  self.w = avg.w;
  self.h = avg.h;
};

/**
 * Calculate the intersection between this and another Box2d.
 * @param {Box2d} box - The other box.
 * @return {number} - The intersection between the two boxes.
 */
Box2d.prototype.intersection = function(box) {
  let self = this;
  let b1x1 = Math.min(self.x, self.x + self.w);
  let b1x2 = Math.max(self.x, self.x + self.w);

  let b1y1 = Math.min(self.y, self.y + self.h);
  let b1y2 = Math.max(self.y, self.y + self.h);

  let b2x1 = Math.min(box.x, box.x + box.w);
  let b2x2 = Math.max(box.x, box.x + box.w);

  let b2y1 = Math.min(box.y, box.y + box.h);
  let b2y2 = Math.max(box.y, box.y + box.h);

  let ix1 = Math.max(b1x1, b2x1);
  let ix2 = Math.min(b1x2, b2x2);
  let iy1 = Math.max(b1y1, b2y1);
  let iy2 = Math.min(b1y2, b2y2);

  return Math.max(0, (ix2 - ix1) * (iy2 - iy1));
};

Box2d.prototype.union = function(box) {
  let self = this;
  let intersection = self.intersection(box);
  return Math.abs(self.w * self.h) + Math.abs(box.w * box.h) - intersection;
};

/**
 * Converts handle number to the central point of specified resize handle.
 * @param {number} handleNo - The handle number, ranges from 0 to 8.
 * @return {object} - A struct with x and y of the handle's center.
 */
Box2d.prototype._getHandle = function(handleNo) {
  let self = this;
  return [
    function() {return {x: self.x, y: self.y};}, // 0
    function() {return {x: self.x + self.w / 2, y: self.y};}, // 1
    function() {return {x: self.x + self.w, y: self.y};}, // 2
    function() {return {x: self.x + self.w, y: self.y + self.h / 2};}, // 3
    function() {return {x: self.x + self.w, y: self.y + self.h};}, // 4
    function() {return {x: self.x + self.w / 2, y: self.y + self.h};}, // 5
    function() {return {x: self.x, y: self.y + self.h};}, // 6
    function() {return {x: self.x, y: self.y + self.h / 2};}, // 7
  ][handleNo - 1]();
};
