// /* global ImageLabel Rect Vertex UP_RES_RATIO FONT_SIZE */

import {ImageLabel} from './image';
import {Rect, Vertex, UP_RES_RATIO} from './shape';
import {FONT_SIZE} from './utils';
import {newLabel, newRect} from './state';
import {List} from 'immutable';

// Constants
const BoxStates = Object.freeze({
  FREE: 0, RESIZE: 1, MOVE: 2,
});
const MIN_BOX_SIZE = 5;

const INITIAL_HANDLE_NO = 4;

/**
 * 2D box label
 * @param {Sat} sat: context
 * @param {int} id: label id
 * @param {object} optionalAttributes - Optional attributes
 */
export function Box2d(sat, id, optionalAttributes) {
  ImageLabel.call(this, sat, id, optionalAttributes);
  this.rect = new Rect();
  this.state = BoxStates.FREE;

  // attributes
  let mousePos;
  if (optionalAttributes) {
    this.categoryPath = optionalAttributes.categoryPath;
    for (let i = 0; i < this.sat.attributes.length; i++) {
      let attributeName = this.sat.attributes[i].name;
      if (attributeName in optionalAttributes.attributes) {
        this.attributes[attributeName] =
            optionalAttributes.attributes[attributeName];
      }
    }
    mousePos = optionalAttributes.mousePos;
    if (mousePos) {
      this.rect.setRect(mousePos.x,
          mousePos.y, 0, 0);
    }
    if (optionalAttributes.shadow) {
      this.setState(BoxStates.FREE);
    } else {
      this.setState(BoxStates.RESIZE);
    }
  }

  this.selectedShape = this.rect.vertices[INITIAL_HANDLE_NO];

  this.selectedCache = null;
  this.interpolateHandler = this.weightedAvg;
}

Box2d.f = {};

Box2d.f.createLabel = function(labelId) {
  return newLabel({id: labelId, shapes: new List([newRect()])});
};

Box2d.prototype = Object.create(ImageLabel.prototype);

Object.defineProperty(Box2d.prototype, 'x', {
  get: function() {
    return this.rect.x;
  },
});

Object.defineProperty(Box2d.prototype, 'y', {
  get: function() {
    return this.rect.y;
  },
});

Object.defineProperty(Box2d.prototype, 'w', {
  get: function() {
    return this.rect.w;
  },
});

Object.defineProperty(Box2d.prototype, 'h', {
  get: function() {
    return this.rect.h;
  },
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
  if (shape === this.rect) {
    return true;
  }
  for (let v of this.rect.vertices) {
    if (shape === v) {
      return true;
    }
  }
  return false;
};

Box2d.prototype.releaseAsTargeted = function() {
  this.setState(BoxStates.FREE);
  ImageLabel.prototype.releaseAsTargeted.call(this);
};

/**
 * Load label data from a encoded string
 * @param {object} json: json representation of label json.
 */
Box2d.prototype.fromJsonVariables = function(json) {
  this.decodeBaseJsonVariables(json);
  if (json.data) {
    // Data may be missing, e.g. if this json is actually a track.
    //   In that case, this is later rectified by the decodeBaseJson of
    //   VideoSat.
    this.rect.setRect(json.data.x, json.data.y, json.data.w, json.data.h);
  }
};

/**
 * Load label data from export format
 * @param {object} exportFormat: json representation of label json.
 * @return {Box2d} the label loaded by exportFormat
 */
Box2d.prototype.fromExportFormat = function(exportFormat) {
  if (exportFormat['box2d']) {
    let [x1, x2, y1, y2] = [
      exportFormat['box2d'].x1, exportFormat['box2d'].x2,
      exportFormat['box2d'].y1, exportFormat['box2d'].y2];
    this.rect.setRect(x1, y1, x2 - x1, y2 - y1);
    this.categoryPath = exportFormat.category;
    this.attributes = exportFormat.attributes;
    return this;
  }
  return null;
};

/**
 * Encode the label data into a json object.
 * @return {object} - the encoded json object.
 */
Box2d.prototype.toJson = function() {
  let json = this.encodeBaseJson();
  json.data = {
    x: this.rect.x,
    y: this.rect.y,
    w: this.rect.w,
    h: this.rect.h,
  };
  return json;
};

/**
 * Returns the shapes to draw on the hidden canvas when not selected.
 * @return {[Shape]} List of shapes to draw on the hidden canvas
 * when not selected.
 */
Box2d.prototype.defaultHiddenShapes = function() {
  return this.getAllHiddenShapes();
};

/**
 * Returns all hidden shapes.
 * @return {[Shape]} List of all hidden shapes
 */
Box2d.prototype.getAllHiddenShapes = function() {
  return [this.rect].concat(this.rect.vertices);
};

Box2d.prototype.deleteAllHiddenShapes = function() {
  this.rect.delete();
};

/**
 * Draw this bounding box on the canvas.
 * @param {object} mainCtx - HTML canvas context for visible objects.
 */
Box2d.prototype.redrawLabelCanvas = function(mainCtx) {
  // set context font
  mainCtx.font = FONT_SIZE * UP_RES_RATIO + 'px Verdana';
  // draw visible elements
  mainCtx.strokeStyle = this.styleColor();
  this.rect.draw(mainCtx, this.satItem, this.state === BoxStates.RESIZE);

  if (this.isTargeted() || this.hoveredShape) {
    this.rect.drawHandles(mainCtx, this.satItem, this.styleColor(),
        this.hoveredShape);
    this.hoveredShape = null;
  }
  if (this.state === BoxStates.FREE) {
    let tlx = Math.min(this.rect.x, this.rect.x + this.rect.w);
    let tly = Math.min(this.rect.y, this.rect.y + this.rect.h);
    this.drawTag(mainCtx, [tlx, tly]);
  }
};

/**
 * Get whether this bounding box is geometrically valid.
 * @return {boolean} - True if the box is geometrically valid.
 */
Box2d.prototype.shapesValid = function() {
  return !this.isSmall();
};

Box2d.prototype.isSmall = function() {
  let [w, h] = this.satItem.toCanvasCoords([this.rect.w, this.rect.h]);
  return (w < MIN_BOX_SIZE * UP_RES_RATIO || h < MIN_BOX_SIZE * UP_RES_RATIO);
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
    return [
      'nwse-resize', 'ns-resize', 'nesw-resize', 'ew-resize',
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
    // this.satItem.resetHiddenMap(this.getAllHiddenShapes());
    // this.satItem.redrawHiddenCanvas();
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
      startBox.x + weight * (endBox.x - startBox.x),
      startBox.y + weight * (endBox.y - startBox.y),
      startBox.w + weight * (endBox.w - startBox.w),
      startBox.h + weight * (endBox.h - startBox.h),
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
  avg.x = self.x + weight * (box.x - self.x);
  avg.y = self.y + weight * (box.y - self.y);
  avg.w = self.w + weight * (box.w - self.w);
  avg.h = self.h + weight * (box.h - self.h);
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
 * Set this box to have the provided rect.
 * @param {Rect} rect - The provided rect.
 */
Box2d.prototype.setShape = function(rect) {
  this.rect.setRect(rect.x, rect.y, rect.w, rect.h);
};

/**
 * Set this box to be a sized down version of the provided box.
 * @param {Box2d} startBox - The provided box.
 */
Box2d.prototype.shrink = function(startBox) {
  let self = this;
  self.rect.setRect(startBox.x, startBox.y, startBox.w / 2, startBox.h / 2);
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
    if (occupiedShape) {
      this.selectedCache = occupiedShape.copy();
    }

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
    if (!this.shapesValid()) {
      this.satItem.deselectAll();
      return;
    }
  } else if (this.state === BoxStates.MOVE) {
    this.mouseClickPos = null;
  }
  this.satItem.redrawHiddenCanvas();
  this.setState(BoxStates.FREE);
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
    // make moved box within padBox
    let [padBoxX, padBoxY] = this.satItem.toImageCoords(
        [this.satItem.padBox.x, this.satItem.padBox.y]);
    let [padBoxW, padBoxH] = this.satItem.toImageCoords(
        [this.satItem.padBox.w, this.satItem.padBox.h], false);

    dx = Math.min(dx, (padBoxX + padBoxW - this.rect.w) - this.selectedCache.x);
    dx = Math.max(dx, padBoxX - this.selectedCache.x);
    dy = Math.min(dy, (padBoxY + padBoxH - this.rect.h) - this.selectedCache.y);
    dy = Math.max(dy, padBoxY - this.selectedCache.y);

    for (let i = 0; i < this.rect.vertices.length; i++) {
      this.rect.getVertex(i).x = this.selectedCache.getVertex(i).x + dx;
      this.rect.getVertex(i).y = this.selectedCache.getVertex(i).y + dy;
    }
    if (this.parent) {
      this.parent.interpolate(this); // TODO
    }
  }
};

Box2d.prototype.mouseleave = function(e) { // eslint-disable-line
  this.mouseup();
  this.satItem.isMouseDown = false;
};

