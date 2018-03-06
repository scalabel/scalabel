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
  // TODO(Wenqi): Add more methods and variables
}

Box2dImage.prototype = Object.create(SatImage.prototype);

/**
 * 2D box label
 * @param {Sat} sat: context
 * @param {int} id: label id
 */
function Box2d(sat, id) {
  ImageLabel.call(this, sat, id);
  // TODO(Wenqi): Add more methods and variables
}

Box2d.prototype = Object.create(ImageLabel.prototype);

/**
 * Draw this bounding box on the canvas.
 * @param {object} canvas - HTML canvas for visible objects.
 * @param {object} hiddenCanvas - HTML canvas for hidden objects.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param {boolean} resizing - Whether or not this box is being resized.
 * @param{number} hoverBox - ID of the currently hovered over box, or null if
 *   no box hovered over.
 * @param {number} hoverHandle - handle number of the currently hovered handle,
 *   or null if no handle hovered.
 */
Box2d.prototype.redraw = function(canvas, hiddenCanvas, selectedBox, resizing,
 hoverBox, hoverHandle) {
  let self = this;

  // get contexts from canvases
  let ctx = canvas.getContext('2d');
  let hiddenCtx = hiddenCanvas.getContext('2d');

  // draw visible elements
  self.drawBox(ctx, selectedBox, resizing);
  self.drawHandles(ctx, selectedBox, hoverBox, hoverHandle);

  // draw hidden elements
  self.drawHiddenBox(hiddenCtx, selectedBox);
  self.drawHiddenHandles()

}

/**
 * Draw the box part of this bounding box.
 * @param {object} canvas - HTML canvas for visible objects.
 * @param {object} ctx - Canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param {boolean} resizing - Whether or not this box is being resized.
 */
Box2d.prototype.drawBox = function(ctx, selectedBox, resizing) {
  let self = this;
  ctx.save(); // save the canvas context settings
  if (selectedBox && selectedBox != this.id) {
    // if exists selected box and it's not this one, alpha this out
    ctx.globalAlpha = FADED_ALPHA; // TODO: where is FADED_ALPHA?
  }
  if (resizing) {
    ctx.setLineDash([3]); // if box is being resized, use line dashes
  }
  if (self.isSmall()) {
    ctx.strokeStyle = 'rgb(169, 169, 169)' // if box is too small, gray it out
  } else {
    // otherwise use regular color
    ctx.strokeStyle = self.color();
  }
  ctx.lineWidth = LINE_WIDTH; // set line width TODO: where is LINE_WIDTH?
  ctx.strokeRect(self.x1, self.y1, self.w, self.h); // draw the box
  ctx.restore(); // restore the canvas to saved settings
}

/**
 * Draw the resize handles of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param {number} hoverHandle - ID of the currently hovered handle, or null
 */
Box2d.prototype.drawHandles = function(ctx, selectedBox, hoverBox, hoverHandle) {
  let self = this;
  if (selectedBox === self.id) {
    // if this box is selected, draw all its handles
    for (let handleNo = 0; handleNo < 8; handleNo++) {
      self.drawHandle(ctx, handleNo);
    }
  } else if (!selectedBox && hoverBox === self.id && hoverHandle) {
    // else if no selection and a handle is hovered over, draw it
    self.drawHandle(ctx, hoverHandle);
  }
}

/**
 * Draw a specified resize handle of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
Box2d.prototype.drawHandle = function(ctx, handleNo) {
  let self = this;
  ctx.save(); // save the canvas context settings
  let posHandle = self._getHandle(handleNo)
  if (self.isSmall()) {
    ctx.fillStyle = 'rgb(169, 169, 169)';
  } else {
    ctx.fillStyle = self.color();
  }
  ctx.lineWidth = LINE_WIDTH; // TODO: where is LINE_WIDTH?
  ctx.beginPath();
  ctx.arc(posHandle.x, posHandle.y, HANDLE_RADIUS, 0, 2 * Math.PI);
  ctx.fill();
  if (!self.isSmall()) {
    ctx.fillStyle = 'white';
    ctx.lineWidth = OUTLINE_WIDTH; // TODO: where is OUTLINE_WIDTH?
    ctx.stroke();
  }
  ctx.restore(); // restore the canvas to saved settings
}

/**
 * Draw the label tag of this bounding box.
 * @param {object} ctx - Canvas context.
 */
Box2d.prototype.drawTag = function(ctx) {
  let self = this;
  if (!self.isSmall()) {
    ctx.save();
    let words = self.category.split(' ');
    // abbreviate tag as the first 3 chars of the last word
    let abbr = words[words.length - 1].substring(0, 3);
    ctx.fillStyle = self.color();
    ctx.fillRect(self.x1 + 1, self.y1 - TAG_HEIGHT, TAG_WIDTH, TAG_HEIGHT); // TODO: define constants
    ctx.fillStyle = 'rgb(0, 0, 0)';
    ctx.fillText(abbr, self.x1 + 3, self.y1 - 3); // TODO: define these as constants
    ctx.restore();
  }
}

/**
 * Draw the box part of the hidden box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 */
Box2d.prototype.drawHiddenBox = function(hiddenCtx, selectedBox) {
  // only draw if it is not the case that there is another selected box
  let self = this;
  if (!selectedBox || selectedBox === self.id) {
    hiddenCtx.save(); // save the canvas context settings
    hiddenCtx.strokeStyle = self.hiddenColor(8); // 8 represents the box itself
    hiddenCtx.lineWidth = HIDDEN_LINE_WIDTH; // TODO: where is HIDDEN_LINE_WIDTH?
    hiddenCtx.strokeRect(self.x1, self.y1, self.w, self.h); // draw the box
    hiddenCtx.restore(); // restore the canvas to saved settings
  }
}

/**
 * Draw the hidden resize handles of this bounding box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 * @param {number} hoverHandle - ID of the currently hovered handle, or null
 */
Box2d.prototype.drawHiddenHandles = function(hiddenCtx, selectedBox, hoverHandle) {
  let self = this;
  if (selectedBox === self.id) {
    // if this box is selected, draw all its hidden handles
    for (let handleNo = 0; handleNo < 8; handleNo++) {
      self.drawHiddenHandle(hiddenCtx, handleNo);
    }
  } else if (!selectedBox && hoverBox === self.id && hoverHandle) {
    // else if no selection and a handle is hovered over, draw it
    self.drawHiddenHandle(hiddenCtx, hoverHandle);
  }
}

/**
 * Draw a specified hidden resize handle of this bounding box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
Box2d.prototype.drawHiddenHandle = function(hiddenCtx, handleNo) {
  let self = this;
  hiddenCtx.save(); // save the canvas context settings
  let posHandle = self._getHandle(handleNo)
  hiddenCtx.fillStyle = self.hiddenColor(handleNo);
  hiddenCtx.lineWidth = HIDDEN_LINE_WIDTH; // TODO: where is HIDDEN_LINE_WIDTH?
  hiddenCtx.beginPath();
  hiddenCtx.arc(posHandle.x, posHandle.y, HIDDEN_HANDLE_RADIUS, 0, 2 * Math.PI);
  hiddenCtx.fill();
  hiddenCtx.restore(); // restore the canvas to saved settings
}

/**
 * Get whether this bounding box is too small.
 * @return {boolean} - True if the box is too small.
 */
Box2d.prototype.isSmall = function() {
  return Math.min(this.w, this.h) < MIN_BOX_SIZE;
  // TODO: where do final vars like MIN_BOX_SIZE go?
  // TODO: define Box2d variables
}

Box2d.prototype.hiddenColor = function(handle) {

}

/**
 * Converts handle number to the central point of specified resize handle.
 * @param {number} - The number corresponding to 1 of 8 resize handles.
 * @return {object} - A struct with x and y of the handle's center.
 */
Box2d.prototype._getHandle = function(handleNo) {
  let self = this;
  switch(index) {
    case 0:
      return {x: self.x1, y: self.y1};
    case 1:
      return {x: self.x1 + self.w, y: self.y1};
    case 2:
      return {x: self.x1, y: self.y1 + self.h};
    case 3:
      return {x: self.x1 + self.w, y: self.y1 + self.h};
    case 4:
      return {x: self.x1 + self.w / 2, y: self.y1};
    case 5:
      return {x: self.x1, y: self.y1 + self.h / 2};
    case 6:
      return {x: self.x1 + self.w / 2, y: self.y1 + self.h};
    case 7:
      return {x: self.x1 + self.w, y: self.y1 + self.h / 2};
    case 8:
      return; // this shouldn't happen (8 is for the box)
  }
}