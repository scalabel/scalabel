/* global ImageLabel Rect Vertex */
/* exported Box2d */

// Constants
const BoxStates = Object.freeze({
  FREE: 0, RESIZE: 1, MOVE: 2});

const INITIAL_HANDLE_NO = 4;

/**
 * 2D box label
 * @param {Sat} sat: context
 * @param {int} id: label id
 * @param {object} optionalAttributes - Optional attributes
 */
function Box2d(sat, id, optionalAttributes) {
  ImageLabel.call(this, sat, id, optionalAttributes);

  // attributes
  if (optionalAttributes) {
    this.categoryPath = optionalAttributes.categoryPath;
    this.trunc = optionalAttributes.trunc;
    this.occl = optionalAttributes.occl;
  }

  this.rect = new Rect();
  if (optionalAttributes.mousePos) {
    this.rect.setRect(optionalAttributes.mousePos.x,
      optionalAttributes.mousePos.y, 0, 0);
  }
  if (optionalAttributes.shadow) {
    this.setState(BoxStates.FREE);
  } else {
    this.setState(BoxStates.RESIZE);
  }
  this.selectedShape = this.rect.vertices[INITIAL_HANDLE_NO];
  this.satItem.pushToHiddenMap(this.defaultHiddenShapes());

  this.selectedCache = null;
}

Box2d.prototype = Object.create(ImageLabel.prototype);

Object.defineProperty(Box2d.prototype, 'x', {
  get: function() {return this.rect.x;},
});

Object.defineProperty(Box2d.prototype, 'y', {
  get: function() {return this.rect.y;},
});

Object.defineProperty(Box2d.prototype, 'w', {
  get: function() {return this.rect.w;},
});

Object.defineProperty(Box2d.prototype, 'h', {
  get: function() {return this.rect.h;},
});

Box2d.useCrossHair = true;
Box2d.defaultCursorStyle = 'crosshair';
Box2d.useDoubleClick = false;

Box2d.setToolBox = function(satItem) { // eslint-disable-line

};

/**
 * Check whether given index selects this Box2d.
 * @param {Shape} shape: the shape under the mouse.
 * @return {boolean} whether the index selects this Box2d.
 */
Box2d.prototype.selectedBy = function(shape) {
  if (shape === this.rect) return true;
  for (let v of this.rect.vertices) {
    if (shape === v) return true;
  }
  return false;
};

Box2d.prototype.toJson = function() {
  let self = this;
  let json = self.encodeBaseJson();
  json.box2d = {x: self.x, y: self.y, w: self.w, h: self.h};
  // TODO: customizable
  json.attributeValues = {occlusion: self.occl, truncation: self.trunc};
  return json;
};

/**
 * Load label information from json object
 * @param {object} json: JSON representation of this Box2d.
 */
Box2d.prototype.fromJsonVariables = function(json) {
  let self = this;
  self.decodeBaseJsonVariables(json);
  if (json.box2d) {
    self.x = json.box2d.x;
    self.y = json.box2d.y;
    self.w = json.box2d.w;
    self.h = json.box2d.h;
  }
  if (json.attributeValues) {
    self.occl = json.attributeValues.occlusion;
    self.trunc = json.attributeValues.truncation;
  }
};

/**
 * Returns the shapes to draw on the hidden canvas when not selected.
 * @return {[Shape]} List of shapes to draw on the hidden canvas
 * when not selected.
 */
Box2d.prototype.defaultHiddenShapes = function() {
  return [this.rect].concat(this.rect.vertices);
};

/**
 * Draw this bounding box on the canvas.
 * @param {object} mainCtx - HTML canvas context for visible objects.
 */
Box2d.prototype.redrawMainCanvas = function(mainCtx) {
  // go ahead and set context font
  mainCtx.font = '11px Verdana';

  // draw visible elements
  mainCtx.strokeStyle = this.styleColor();
  this.rect.draw(mainCtx, this.satItem, this.state === BoxStates.RESIZE);
  if (this.state === BoxStates.FREE) {
    let tlx = Math.min(this.rect.x, this.rect.x + this.rect.w);
    let tly = Math.min(this.rect.y, this.rect.y + this.rect.h);
    this.drawTag(mainCtx, [tlx, tly]);
  }

  if (this.isTargeted() || this.hoveredShape) {
    this.rect.drawHandles(mainCtx, this.satItem, this.styleColor(),
        this.hoveredShape);
    this.hoveredShape = null;
  }
};

/**
 * Get whether this bounding box is valid.
 * @return {boolean} - True if the box is valid.
 */
Box2d.prototype.isValid = function() {
  return this.rect.isValid();
};

/**
 * Get the cursor style for a specified handle number.
 * @param {Shape} shape - The Shape object that determines the cursor
 * @return {string} - The cursor style string.
 */
Box2d.prototype.getCursorStyle = function(shape) {
  if (this.state === BoxStates.MOVE || shape instanceof Rect) {
    return 'move';
  }
  let handleNo = this.rect.getHandleNo(shape);
  if (handleNo < 0) {
    return this.defaultCursorStyle;
  }
  if (this.state === BoxStates.RESIZE) {
    return 'crosshair';
  } else {
    return ['nwse-resize', 'ns-resize', 'nesw-resize', 'ew-resize',
      'nwse-resize', 'ns-resize', 'nesw-resize', 'ew-resize'][handleNo];
  }
};
/**
 * Function to set the current state.
 * @param {number} state - The state to set to.
 */
Box2d.prototype.setState = function(state) {
  if (state === BoxStates.FREE) {
    this.state = state;
    this.selectedShape = this.rect;
    this.selectedCache = null;
  } else if (state === BoxStates.RESIZE) {
    this.state = state;
  } else if (state === BoxStates.MOVE) {
    this.state = state;
  }
};

/**
 * Set this box to be the weighted average of the two provided boxes.
 * @param {Box2d} startBox - The first box.
 * @param {Box2d} endBox - The second box.
 * @param {number} weight - The weight, b/w 0 and 1, higher corresponds to
 closer to endBox.
 */
Box2d.prototype.weightAvg = function(startBox, endBox, weight) {
  let self = this;
  self.rect.setRect(
      startBox.x + weight*(endBox.x - startBox.x),
      startBox.y + weight*(endBox.y - startBox.y),
      startBox.w + weight*(endBox.w - startBox.w),
      startBox.h + weight*(endBox.h - startBox.h)
  );
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
  self.rect.setRect(avg.x, avg.y, avg.w, avg.h);
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

Box2d.prototype.mousedown = function(e) {
  let mousePos = this.satItem.getMousePos(e);
  // TODO: Wenqi
  // traffic light color
  if (this.state === BoxStates.FREE) {
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    let occupiedLabel = this.satItem.getLabelOfShape(occupiedShape);
    this.selectedCache = occupiedShape.copy();

    if (occupiedLabel && occupiedLabel.id === this.id) {
      if (occupiedShape instanceof Vertex) {
        this.selectedShape = occupiedShape;
        this.setState(BoxStates.RESIZE);
      } else if (occupiedShape instanceof Rect) {
        this.setState(BoxStates.MOVE);
        this.mouseClickPos = mousePos;
      }
    } else {
      this.satItem.deselectAll();
    }
  } else if (this.state === BoxStates.RESIZE) {
    // just entered RESIZE state
    this.rect.setRect(mousePos.x, mousePos.y, 0, 0);
  }
};

Box2d.prototype.mouseup = function() {
  if (this.state === BoxStates.RESIZE) {
    this.selectedShape = null;
    let x = Math.min(this.rect.getVertex(0).x, this.rect.getVertex(4).x);
    let y = Math.min(this.rect.getVertex(0).y, this.rect.getVertex(4).y);
    let w = Math.abs(this.rect.getVertex(0).x - this.rect.getVertex(4).x);
    let h = Math.abs(this.rect.getVertex(0).y - this.rect.getVertex(4).y);

    this.rect.setRect(x, y, w, h);
    if (!this.isValid()) {
      if (this.selectedCache) {
        this.rect.setRect([this.selectedCache.x, this.selectedCache.y,
          this.selectedCache.w, this.selectedCache.h]);
      } else {
        this.satItem.deselectAll();
        this.delete();
      }
    }
  } else if (this.state === BoxStates.MOVE) {
    this.mouseClickPos = null;
  }

  this.setState(BoxStates.FREE);

  // if parent label, make this the selected label in all other SatImages
  let currentItem = this.sat.currentItem;
  if (this.parent) {
    currentItem = currentItem.previousItem();
    let currentLabel = this.sat.labelIdMap[this.previousLabelId];
    while (currentItem) {
      currentItem.selectedLabel = currentLabel;
      currentItem.currHandle = currentItem.selectedLabel.INITIAL_HANDLE;
      if (currentLabel) {
        currentLabel = this.sat.labelIdMap[currentLabel.previousLabelId];
        // TODO: make both be functions, not attributes
      }
      currentItem = currentItem.previousItem();
    }
    currentItem = this.sat.currentItem.nextItem();
    currentLabel = this.sat.labelIdMap[this.nextLabelId];
    while (currentItem) {
      currentItem.selectedLabel = currentLabel;
      currentItem.currHandle = currentItem.selectedLabel.INITIAL_HANDLE;
      if (currentLabel) {
        currentLabel = this.sat.labelIdMap[currentLabel.nextLabelId];
      }
      currentItem = currentItem.nextItem();
    }
  }
};

Box2d.prototype.mousemove = function(e) {
  let mousePos = this.satItem.getMousePos(e);

  // handling according to state
  if (this.state === BoxStates.RESIZE) {
    let movedVertex = this.selectedShape;
    let handleNo = this.rect.getHandleNo(movedVertex);
    let oppHandleNo = this.rect.oppositeHandleNo(handleNo);
    let oppVertex = this.rect.vertices[oppHandleNo];
    if (handleNo % 2 === 0) {
      // move a vertex
      movedVertex.xy = [mousePos.x, mousePos.y];

      this.rect.getVertex(handleNo + 2).xy = [movedVertex.x, oppVertex.y];
      this.rect.getVertex(handleNo - 2).xy = [oppVertex.x, movedVertex.y];
      // update midpoints
      this.rect.updateMidpoints();
    } else {
      // move a midpoint
      if (handleNo === 1 || handleNo === 5) {
        // vertical
        movedVertex.y = mousePos.y;
        this.rect.getVertex(handleNo + 1).y = movedVertex.y;
        this.rect.getVertex(handleNo - 1).y = movedVertex.y;
        // update midpoints
        this.rect.updateMidpoints();
      } else if (handleNo === 3 || handleNo === 7) {
        // horizontal
        movedVertex.x = mousePos.x;
        this.rect.getVertex(handleNo + 1).x = movedVertex.x;
        this.rect.getVertex(handleNo - 1).x = movedVertex.x;
        // update midpoints
        this.rect.updateMidpoints();
      }
    }
    if (this.parent) {
      this.parent.interpolate(this); // TODO
    }
  } else if (this.state === BoxStates.MOVE) {
    let dx = mousePos.x - this.mouseClickPos.x;
    let dy = mousePos.y - this.mouseClickPos.y;
    for (let i = 0; i < this.rect.vertices.length; i++) {
      this.rect.getVertex(i).x = this.selectedCache.getVertex(i).x + dx;
      this.rect.getVertex(i).y = this.selectedCache.getVertex(i).y + dy;
    }
    if (this.parent) {
      this.parent.interpolate(this); // TODO
    }
  }
};
