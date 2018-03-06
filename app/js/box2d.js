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
 */
Box2d.prototype.redraw = function(canvas, hiddenCanvas, selectedBox, resizing) {
  let self = this;

  // get contexts from canvases
  let ctx = canvas.getContext('2d');
  let hiddenCtx = hiddenCanvas.getContext('2d');

  // draw visible elements
  self.drawBox(ctx, selectedBox, resizing);

  // draw hidden elements
  self.drawHiddenBox(hiddenCtx, selectedBox)

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
  ctx.save(); // save the canvas context settings
  if (selectedBox && selectedBox != this.id) {
    // if exists selected box and it's not this one, alpha this out
    ctx.globalAlpha = FADED_ALPHA; // TODO: where is FADED_ALPHA?
  }
  if (resizing) {
    ctx.setLineDash([3]); // if box is being resized, use line dashes
  }
  if (this.isSmall()) {
    ctx.strokeStyle = 'rgb(169, 169, 169)' // if box is too small, gray it out
  } else {
    // otherwise use regular color
    ctx.strokeStyle = this.color();
  }
  ctx.lineWidth = LINE_WIDTH; // set line width TODO: where is LINE_WIDTH?
  ctx.strokeRect(this.x1, this.y1, this.w, this.h); // draw the box
  ctx.restore(); // restore the canvas to saved settings
}

Box2d.prototype.drawHandles = function() {

}

Box2d.prototype.drawHandleOutlines = function() {

}

Box2d.prototype.drawLabel = function() {

}

/**
 * Draw the box part of the hidden box.
 * @param {object} hiddenCtx - Hidden canvas context.
 * @param {number} selectedBox - ID of the currently selected box, or null if
 *   no box selected.
 */
Box2d.prototype.drawHiddenBox = function(hiddenCtx, selectedBox) {
  // only draw if it is not the case that there is another selected box
  if (!selectedBox || selectedBox === this.id) {
    hiddenCtx.save(); // save the canvas context settings
    hiddenCtx.strokeStyle = this.hiddenColor(8); // 8 represents the box itself
    hiddenCtx.lineWidth = HIDDEN_LINE_WIDTH; // TODO: where is HIDDEN_LINE_WIDTH?
    hiddenCtx.strokeRect(this.x1, this.y1, this.w, this.h); // draw the box
    hiddenCtx.restore(); // restore the canvas to saved settings
  }
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