import {SatImage, ImageLabel} from './image';
import {
  Polygon, Path, Vertex,
  VertexTypes, GRAYOUT_COLOR, SELECT_COLOR,
  LINE_WIDTH, OUTLINE_WIDTH, ALPHA_HIGH_FILL,
  ALPHA_LOW_FILL, ALPHA_LINE, UP_RES_RATIO,
} from './shape';
import {rgba, FONT_SIZE} from './utils';

// constants
let SegStates = Object.freeze({
  FREE: 0, DRAW: 1, RESIZE: 2, QUICK_DRAW: 3, LINK: 4, MOVE: 5,
});

/**
 * 2D segmentation label
 * @param {Sat} sat: context
 * @param {int} id: label id
 * @param {object} optionalAttributes - Optional attributes
 */
export function Seg2d(sat, id, optionalAttributes) {
  ImageLabel.call(this, sat, id);

  this.polys = [];
  this.selectedShape = null;
  this.quickdrawCache = {};
  this.state = SegStates.FREE;
  this.selectedShape = null;
  this.interpolateHandler = this.identityInterpolation;

  let mousePos;
  if (optionalAttributes) {
    mousePos = optionalAttributes.mousePos;
    this.categoryPath = optionalAttributes.categoryPath;
    for (let i = 0; i < this.sat.attributes.length; i++) {
      let attributeName = this.sat.attributes[i].name;
      if (optionalAttributes.attributes && attributeName
          in optionalAttributes.attributes) {
        this.attributes[attributeName] =
            optionalAttributes.attributes[attributeName];
      }
    }
  }

  if (mousePos) {
    // if mousePos given, start drawing
    // set label type
    this.setState(SegStates.DRAW);
    this.initDrawing(mousePos);
  }
}

Seg2d.prototype = Object.create(ImageLabel.prototype);

Seg2d.useCrossHair = false;
Seg2d.defaultCursorStyle = 'default';
Seg2d._useDoubleClick = true;
Seg2d.useDoubleClick = true;
Seg2d.allowsLinkingWithinFrame = true;
Seg2d.closed =
    document.getElementById('label_type').innerHTML === 'segmentation';

Seg2d.prototype.initDrawing = function(mousePos) {
  if (Seg2d.closed) {
    this.newPoly = new Polygon();
    this.tempPoly = new Polygon(-1);
  } else {
    this.newPoly = new Path();
    this.tempPoly = new Path(-1);
  }
  let occupiedShape = this.satItem.getOccupiedShape(mousePos);
  if (occupiedShape instanceof Vertex) {
    this.newPoly.pushVertex(occupiedShape);
    this.tempPoly.pushVertex(occupiedShape);
    this.satItem.pushToHiddenMap([occupiedShape]);
    this.latestSharedVertex = occupiedShape;
  } else {
    let firstVertex = new Vertex(mousePos.x, mousePos.y);
    this.newPoly.pushVertex(firstVertex);
    this.tempPoly.pushVertex(firstVertex);
    this.satItem.pushToHiddenMap([firstVertex]);
    this.latestSharedVertex = null;
  }

  this.tempVertex = new Vertex(mousePos.x, mousePos.y,
      VertexTypes.VERTEX, -1);
  this.tempPoly.pushVertex(this.tempVertex);
  this.selectedShape = null;
};

Seg2d.prototype.completeDrawing = function() {
  this.newPoly.endPath();
  if (this.newPoly.isValidShape()) {
    this.addShape(this.newPoly);
    if (this.polyBuffer) {
      for (let poly of this.polyBuffer) {
        poly.delete();
      }
    }
  } else if (this.polyBuffer) {
    for (let poly of this.polyBuffer) {
      this.polys.push(poly);
    }
  }
  this.polyBuffer = null;
  this.tempVertex.delete();
  this.tempPoly.delete();
  this.tempVertex = null;
  this.tempPoly = null;
  this.setState(SegStates.FREE);
  this.selectedShape = this.newPoly;
  if (this.parent) {
    this.parent.interpolate(this);
  }
};

Seg2d.prototype.identityInterpolation = function(startLabel,
      startIndex, priorKeyFrameIndex, nextKeyFrameIndex) { // eslint-disable-line
  this.deleteAllPolyline();
  for (let poly of startLabel.polys) {
    this.addShape(poly.copy());
  }
  this.setState(SegStates.FREE);
};

Seg2d.prototype.addShape = function(poly) {
  this.polys = this.polys.concat(poly);
};

/**
 * Split a given polyline from this Seg2d label,
 * and create a new Seg2d label for it.
 * @param {Polyline} poly - the polyline
 */
Seg2d.prototype.splitShape = function(poly) {
  for (let i = 0; i < this.polys.length; i++) {
    if (this.polys[i] === poly) {
      this.polys.splice(i, 1);
      // create a Seg2d label for this polygon
      let label = this.sat.newLabel({
        categoryPath: this.categoryPath, attributes: this.attributes,
        mousePos: null,
      });
      label.addShape(poly);
      label.setState(SegStates.FREE);
      label.releaseAsTargeted();
      break;
    }
  }
};

/**
 * Delete a given polyline from this Seg2d label.
 * @param {Polyline} poly - the polyline
 */
Seg2d.prototype.deletePolyline = function(poly) {
  for (let i = 0; i < this.polys.length; i++) {
    if (this.polys[i] === poly) {
      this.polys[i].delete();
      this.polys.splice(i, 1);
      break;
    }
  }
};

/**
 * Delete all polylines from this Seg2d label.
 */
Seg2d.prototype.deleteAllPolyline = function() {
  for (let i = 0; i < this.polys.length; i++) {
    this.polys[i].delete();
  }
  this.polys = [];
};

Seg2d.prototype.setAsTargeted = function() {
  this.targeted = true;
  if (this.satItem.active && this.state === SegStates.FREE) {
    let shapes = [];
    for (let label of this.satItem.labels) {
      if (label.valid) {
        shapes = shapes.concat(label.defaultHiddenShapes());
        if (label === this) {
          shapes = shapes.concat(label.getVertices());
          shapes = shapes.concat(this.getControlPoints());
        }
      }
    }
    this.satItem.resetHiddenMap(shapes);
  }
};

/**
 * Function to set the current state.
 * @param {number} state - The state to set to.
 */
Seg2d.prototype.setState = function(state) {
  if (state === SegStates.FREE) {
    // clean up buttons
    if (Seg2d.closed) {
      if (this.state === SegStates.QUICK_DRAW) {
        this.resetQuickdrawButton();
      }
      // stop incompletely drawn labels
      if ((this.state === SegStates.DRAW ||
          this.state === SegStates.QUICK_DRAW) &&
          this.polys.length === 0) {
        this.delete();
        return;
      }
    }

    this.state = state;
    this.selectedShape = null;

    // reset hiddenMap with polygons and handles of this
    if (this.satItem.active) {
      let shapes = [];
      for (let label of this.satItem.labels) {
        if (label.valid) {
          shapes = shapes.concat(label.defaultHiddenShapes());
          if (label === this) {
            shapes = shapes.concat(label.getVertices());
            shapes = shapes.concat(this.getControlPoints());
          }
        }
      }
      this.satItem.resetHiddenMap(shapes);
      this.satItem.redrawHiddenCanvas();
    }
    this.newPoly = null;
    this.tempPoly = null;
    this.tempVertex = null;
  } else if (state === SegStates.DRAW) {
    // reset hiddenMap with all existing vertices
    let shapes = [];
    for (let label of this.satItem.labels) {
      if (label.valid) {
        shapes = shapes.concat(label.getVertices());
      }
    }
    // draw once for each shape
    shapes = Array.from(new Set(shapes));

    if (this.state === SegStates.QUICK_DRAW) {
      this.satItem.pushToHiddenMap(shapes);
    } else {
      this.satItem.resetHiddenMap(shapes);
    }
    this.satItem.redrawHiddenCanvas();
    this.state = state;
  } else if (state === SegStates.RESIZE) {
    this.state = state;
  } else if (state === SegStates.QUICK_DRAW) {
    this.quickdrawCache.tempPolyCache = this.tempPoly;
    this.state = state;
    let shapes = [];
    for (let label of this.satItem.labels) {
      if (label.valid) {
        shapes = shapes.concat(label.getPolygons());
      }
    }
    this.satItem.pushToHiddenMap(shapes);
    // draw once for each shape
    this.satItem._hiddenMap.removeDuplicate();
    this.satItem.redrawHiddenCanvas();
  } else if (state === SegStates.MOVE && Seg2d.closed) {
    // Only polygons can be moved, not lanes
    this.state = state;
  }
};

/**
 * Function to set the tool box of Seg2d.
 * @param {object} satItem - the SatItem object.
 */
Seg2d.setToolBox = function(satItem) {
  if (Seg2d.closed) {
    satItem.isLinking = false;
    if (document.getElementById('link_btn')) {
      document.getElementById('link_btn').onclick = function() {
        satItem.linkHandler();
      };
    }
    document.getElementById('quickdraw_btn').onclick = function() {
      if (satItem.selectedLabel) {
        satItem.selectedLabel.handleQuickdraw();
      }
    };
  }
};

/**
 * Convert midpoint to a new vertex
 * @param {Vertex} pt: midpoint to be converted to a vertex
 * @return {Vertex} the converted vertex
 */
Seg2d.prototype.midpointToVertex = function(pt) {
  let v1;
  let v2;
  let vertex;
  for (let poly of this.polys) {
    if (poly.control_points.indexOf(pt) >= 0) {
      let edgeIndex = poly.getEdgeIndexWithControlPoint(pt);
      vertex = poly.midpointToVertexWithEdgeIndex(edgeIndex);
      [v1, v2] = [poly.edges[edgeIndex].src, poly.edges[edgeIndex + 1].dest];
      break;
    }
  }
  // convert every shared edge
  for (let label of this.satItem.labels) {
    for (let poly of label.polys) {
      poly.convertMidpointToKnownVertexBetween(vertex, v1, v2);
    }
  }
  return vertex;
};

/**
 * Convert midpoint to bezier curve control points
 * @param {Vertex} pt: midpoint to be converted to a vertex
 */
Seg2d.prototype.midpointToBezierControl = function(pt) {
  let v1;
  let v2;
  let c1;
  let c2;
  let convertedPoly;
  for (let poly of this.polys) {
    if (poly.control_points.indexOf(pt) >= 0) {
      let edgeIndex = poly.getEdgeIndexWithControlPoint(pt);
      [c1, c2] = poly.midpointToBezierControlWithEdgeIndex(edgeIndex);
      [v1, v2] = [poly.edges[edgeIndex].src, poly.edges[edgeIndex].dest];
      convertedPoly = poly;
      break;
    }
  }
  // convert every shared edge
  for (let label of this.satItem.labels) {
    for (let poly of label.polys) {
      if (poly !== convertedPoly) {
        poly.convertMidpointToKnownBezierControlBetween(c1, c2, v1, v2);
      }
    }
  }
};

/**
 * Convert two bezier curve control points to a midpoint
 * @param {Vertex} pt: bezier control point to be converted to a vertex
 */
Seg2d.prototype.bezierControlToMidpoint = function(pt) {
  let v1;
  let v2;
  let midpoint;
  let convertedPoly;
  for (let poly of this.polys) {
    if (poly.control_points.indexOf(pt) >= 0) {
      let edgeIndex = poly.getEdgeIndexWithControlPoint(pt);
      midpoint = poly.bezierControlToMidpointWithEdgeIndex(edgeIndex);
      [v1, v2] = [poly.edges[edgeIndex].src, poly.edges[edgeIndex].dest];
      convertedPoly = poly;
      break;
    }
  }
  // convert every shared edge
  for (let label of this.satItem.labels) {
    for (let poly of label.polys) {
      if (poly !== convertedPoly) {
        poly.convertBezierControlToKnownMidpointBetween(midpoint, v1, v2);
      }
    }
  }
};

SatImage.prototype.startLinking = function() {
  let button = document.getElementById('link_btn');
  button.innerHTML = 'Finish Linking';
  button.style.backgroundColor = 'lightgreen';

  let cat = this.catSel.options[this.catSel.selectedIndex].innerHTML;
  if (!this.selectedLabel) {
    let attributes = {};
    for (let i = 0; i < this.sat.attributes.length; i++) {
      if (this.sat.attributes[i].toolType === 'switch') {
        attributes[this.sat.attributes[i].name] = false;
      } else if (this.sat.attributes[i].toolType === 'list') {
        attributes[this.sat.attributes[i].name] = [
          0,
          this.sat.attributes[i].values[0]];
      }
    }
    this.selectedLabel = this.sat.newLabel({
      categoryPath: cat, occl: false,
      trunc: false, mousePos: null,
    });
    this.selectedLabel.setAsTargeted();
  }
  this.selectedLabel.linkHandler();
  this.isLinking = true;
  this.resetHiddenMapToDefault();
  this.redrawHiddenCanvas();
};

SatImage.prototype.endLinking = function() {
  let button = document.getElementById('link_btn');
  this.isLinking = false;
  button.innerHTML = 'Link';
  button.style.backgroundColor = 'white';
  this.updateLabelCount();
  this.selectedLabel.linkHandler();
};

/**
 * Link button handler
 */
SatImage.prototype.linkHandler = function() {
  if (!this.isLinking) {
    this.startLinking();
  } else {
    this.endLinking();
  }
};

Seg2d.prototype.linkHandler = function() {
  if (this.satItem.isLinking) {
    this.setState(SegStates.FREE);
    if (this.polys.length < 1) {
      this.delete();
    }
  }
};

Seg2d.prototype.handleQuickdraw = function() {
  if (this.state === SegStates.DRAW) {
    // s switch to quick draw
    let button = document.getElementById('quickdraw_btn');
    button.innerHTML = 'Select Target Polygon';
    button.style.backgroundColor = 'lightgreen';
    this.setState(SegStates.QUICK_DRAW);
  } else if (this.state === SegStates.QUICK_DRAW) {
    // press s again to quit quick draw
    if (this.quickdrawCache.targetPoly) { // must be before state transition
      this.quickdrawCache.targetSeg2d.releaseAsTargeted();
    }
    if (this.newPoly.vertices.length > 1
        && this.quickdrawCache.endVertex
        && this.quickdrawCache.endVertex.id
        === this.newPoly.vertices[0].id) {
      // if occupied object the 1st vertex, close path
      this.tempPoly.popVertex();
      this.newPoly.endPath();

      if (this.newPoly.isValidShape()) {
        this.addShape(this.newPoly);
        if (this.polyBuffer) {
          for (let poly of this.polyBuffer) {
            poly.delete();
          }
        }
      } else if (this.polyBuffer) {
        for (let poly of this.polyBuffer) {
          this.polys.push(poly);
        }
      }
      this.polyBuffer = null;
      this.tempVertex.delete();
      this.tempPoly.delete();
      this.tempVertex = null;
      this.tempPoly = null;

      this.endQuickDraw();
      this.setState(SegStates.FREE);
      this.selectedShape = this.newPoly;
    } else {
      this.endQuickDraw();
      this.setState(SegStates.DRAW);
    }
  }
};

Seg2d.prototype.endQuickDraw = function() {
  // push the final path to tempPoly and newPoly
  this.tempPoly = this.quickdrawCache.tempPolyCache;
  if (this.quickdrawCache.endVertex) {
    this.tempPoly.popVertex();
    this.tempPoly.pushPath(
        this.quickdrawCache.targetPoly,
        this.quickdrawCache.startVertex,
        this.quickdrawCache.endVertex, this.quickdrawCache.longPath, true);
    this.newPoly.pushPath(
        this.quickdrawCache.targetPoly,
        this.quickdrawCache.startVertex,
        this.quickdrawCache.endVertex, this.quickdrawCache.longPath);
    if (this.quickdrawCache.endVertex === this.newPoly.vertices[0]) {
      this.completeDrawing();
    } else {
      this.tempPoly.pushVertex(this.tempVertex);
      this.setState(SegStates.DRAW);
    }
  } else if (this.quickdrawCache.startVertex) {
    if (this.quickdrawCache.startVertex === this.newPoly.vertices[0]) {
      this.newPoly.endPath();
      this.setState(SegStates.FREE);
    } else {
      this.tempPoly.popVertex();
      this.tempPoly.pushVertex(this.quickdrawCache.startVertex);
      this.tempPoly.pushVertex(this.tempVertex);
      this.newPoly.pushVertex(this.quickdrawCache.startVertex);
      this.setState(SegStates.DRAW);
    }
  } else if (this.quickdrawCache.targetPoly) {
    this.setState(SegStates.DRAW);
  } else {
    this.setState(SegStates.DRAW);
  }
  if (this.quickdrawCache.targetSeg2d) {
    this.quickdrawCache.targetSeg2d.releaseAsTargeted();
  }
  this.quickdrawCache = {};
  this.resetQuickdrawButton();
};

Seg2d.prototype.resetQuickdrawButton = function() {
  let quickdrawButton = document.getElementById('quickdraw_btn');
  quickdrawButton.innerHTML = 'Quick-draw';
  quickdrawButton.style.backgroundColor = 'white';
};

/**
 * Load label data from a encoded string
 * @param {object} json: json representation of label.
 */
Seg2d.prototype.fromJsonVariables = function(json) {
  this.decodeBaseJsonVariables(json);
  this.polys = [];
  if (json.data && json.data.polys) {
    for (let polyJson of json.data.polys) {
      if (Seg2d.closed) {
        this.addShape(Polygon.fromJson(polyJson));
      } else {
        this.addShape(Path.fromJson(polyJson));
      }
    }
  }
};

/**
 * Load label data from export format
 * @param {object} exportFormat: json representation of label json.
 * @return {Box2d} the label loaded by exportFormat
 */
Seg2d.prototype.fromExportFormat = function(exportFormat) {
  if (exportFormat.poly2d && exportFormat.poly2d.length > 0) {
    for (let poly of exportFormat.poly2d) {
      if (poly.closed) {
        this.addShape(Polygon.fromExportFormat(poly));
      } else {
        this.addShape(Path.fromExportFormat(poly));
      }
    }
    this.categoryPath = exportFormat.category;
    this.attributes = exportFormat.attributes;
    this.keyframe = exportFormat.manualShape;
    return this;
  }
  return null;
};

/**
 * Encode the label data into a json object.
 * @return {object} - the encoded json object.
 */
Seg2d.prototype.toJson = function() {
  let json = this.encodeBaseJson();
  let polysJson = [];
  for (let poly of this.polys) {
    polysJson = polysJson.concat(poly.toJson());
  }
  json.data = {
    closed: Seg2d.closed,
    polys: polysJson,
  };
  return json;
};

/**
 * Check whether given index selects this Seg2d.
 * @param {Shape} shape: the shape under the mouse.
 * @return {boolean} whether the index selects this Seg2d.
 */
Seg2d.prototype.selectedBy = function(shape) {
  if (this.polys.length < 1 || shape === null) {
    return false;
  }

  for (let poly of this.polys) {
    // selected by poly
    if (shape === poly) {
      return true;
    }
    // selected by vertex
    for (let pt of poly.vertices) {
      if (shape === pt) {
        return true;
      }
    }
    // selected by control point
    for (let pt of poly.control_points) {
      if (shape === pt) {
        return true;
      }
    }
  }

  return false;
};

Seg2d.prototype.releaseAsTargeted = function() {
  ImageLabel.prototype.releaseAsTargeted.call(this);
};

Seg2d.prototype.deactivate = function() {
  if (this.satItem.isLinking) {
    this.satItem.endLinking();
  }
  if (this.state !== SegStates.FREE) {
    this.setState(SegStates.FREE);
  }
};

Seg2d.prototype.allowsLeavingCurrentItem = function() {
  return this.state !== SegStates.QUICK_DRAW;
};

/**
 * Returns the shapes to draw on the hidden canvas when not selected.
 * @return {[Shape]} List of shapes to draw on the hidden canvas
 * when not selected.
 */
Seg2d.prototype.defaultHiddenShapes = function() {
  return this.getPolygons();
};

/**
 * Returns all polygons of this Seg2d object.
 * @return {[Polygon]} List of polygons of this Seg2d object.
 */
Seg2d.prototype.getPolygons = function() {
  return this.polys;
};

/**
 * Returns all vertices of this Seg2d object.
 * @return {[Vertex]} List of vertices of this Seg2d object.
 */
Seg2d.prototype.getVertices = function() {
  let vertices = [];
  for (let poly of this.polys) {
    vertices = vertices.concat(poly.vertices);
  }
  // draw once for each shape
  vertices = Array.from(new Set(vertices));
  return vertices;
};

/**
 * Returns all vertices of this Seg2d object.
 * @return {[Vertex]} List of vertices of this Seg2d object.
 */
Seg2d.prototype.getEdges = function() {
  let edges = [];
  for (let poly of this.polys) {
    edges = edges.concat(poly.edges);
  }
  // draw once for each shape
  edges = Array.from(new Set(edges));
  return edges;
};

/**
 * Returns all control points of this Seg2d object.
 * @return {[Vertex]} List of control points of this Seg2d object.
 */
Seg2d.prototype.getControlPoints = function() {
  let controlPoints = [];
  for (let poly of this.polys) {
    controlPoints = controlPoints.concat(poly.control_points);
  }
  return controlPoints;
};

/**
 * Returns all shapes of this Seg2d object.
 * @return {[Shape]} List of all hidden shapes of this Seg2d object.
 */
Seg2d.prototype.getAllHiddenShapes = function() {
  let shapes = [];
  shapes = shapes.concat(this.getPolygons());
  shapes = shapes.concat(this.getVertices());
  shapes = shapes.concat(this.getControlPoints());
  if (this.state === SegStates.DRAW) {
    shapes = shapes.concat(this.newPoly);
    shapes = shapes.concat(this.newPoly.vertices);
  }
  return shapes;
};

Seg2d.prototype.deleteAllHiddenShapes = function() {
  for (let poly of this.polys) {
    poly.delete();
  }
};

/**
 * Draw this bounding box on the canvas.
 * @param {object} mainCtx - HTML canvas context for visible objects.
 */
Seg2d.prototype.redrawLabelCanvas = function(mainCtx) {
  // set context font
  mainCtx.font = FONT_SIZE * UP_RES_RATIO + 'px Verdana';
  mainCtx.save(); // save the canvas context settings
  let styleColor = this.styleColor();
  mainCtx.strokeStyle = styleColor;
  this.setPolygonLine(mainCtx);
  this.setPolygonFill(mainCtx);

  if ((this.state === SegStates.DRAW ||
      this.state === SegStates.QUICK_DRAW) && this.tempPoly) {
    this.tempPoly.draw(mainCtx, this.satItem,
      this.isTargeted() && this.state !== SegStates.DRAW
        && this.state !== SegStates.QUICK_DRAW, false);
    this.tempPoly.drawHandles(mainCtx, this.satItem, styleColor,
      this.hoveredShape, false);
  } else if (this.polys.length > 0) {
    for (let poly of this.polys) {
      let drawDashWhenTargeted = this.isTargeted()
          && this.state !== SegStates.DRAW
          && this.state !== SegStates.QUICK_DRAW;
      let polyHovered = this.polys.indexOf(this.hoveredShape) >= 0;
      poly.draw(mainCtx, this.satItem,
          drawDashWhenTargeted || polyHovered,
          this.getRoot().linkTarget || (this.sat.linkingTrack &&
            this.sat.linkingTrack.id === this.getRoot().id));
      if (this.isTargeted() || polyHovered) {
        poly.drawHandles(mainCtx,
            this.satItem, styleColor, this.hoveredShape, true);
      }
    }
    mainCtx.fillStyle = this.styleColor();
    if (!this.satItem.soloMode) {
      this.drawTag(mainCtx, this.polys[0].centroidCoords());
    }
  }

  mainCtx.restore();
};

Seg2d.prototype.setPolygonLine = function(ctx) {
  // set line width
  ctx.lineWidth = OUTLINE_WIDTH;
  if (this.isTargeted()) {
    ctx.lineWidth = LINE_WIDTH;
  }
};

Seg2d.prototype.setPolygonFill = function(ctx) {
  // set color and alpha
  if (this.state !== SegStates.DRAW
      && this.state !== SegStates.QUICK_DRAW
      && this.isTargeted()) {
    ctx.fillStyle = rgba(SELECT_COLOR, ALPHA_HIGH_FILL);
  } else {
    ctx.fillStyle = this.styleColor(ALPHA_LOW_FILL);
  }

  if (this.state === SegStates.DRAW || this.state === SegStates.QUICK_DRAW) {
    ctx.strokeStyle = this.styleColor(ALPHA_LINE);
    ctx.fillStyle = this.styleColor(ALPHA_LOW_FILL);
  } else if (this.state === SegStates.RESIZE) {
    if (this.shapesValid()) {
      ctx.strokeStyle = this.styleColor(ALPHA_LINE);
      ctx.fillStyle = this.styleColor(ALPHA_LOW_FILL);
    } else {
      ctx.strokeStyle = rgba(
          GRAYOUT_COLOR, ALPHA_LINE);
      ctx.fillStyle = rgba(GRAYOUT_COLOR, ALPHA_LOW_FILL);
    }
  }
};

/**
 * Get whether this Seg2d object is geometrically valid.
 * @return {boolean} - True if the Seg2d object is geometrically valid.
 */
Seg2d.prototype.shapesValid = function() {
  if (this.polys.length < 1) {
    return false;
  }
  for (let poly of this.polys) {
    if (!poly.isValidShape()) {
      return false;
    }
  }
  return true;
};

/**
 * Get the cursor style for a specified handle number.
 * @param {Shape} shape - The shape that determines the cursor style.
 * @return {string} - The cursor style string.
 */
Seg2d.prototype.getCursorStyle = function(shape) {
  if (shape instanceof Polygon && !this.satItem.isLinking &&
      !this.sat.linkingTrack && this.satItem.isDown('M') &&
      this.satItem.selectedLabel && this.satItem.selectedLabel.id === this.id) {
    return 'move';
  }
  return Seg2d.defaultCursorStyle;
};

Seg2d.prototype.mousedown = function(e) {
  let mousePos = this.satItem.getMousePos(e);

  let occupiedShape = this.satItem.getOccupiedShape(mousePos);
  if ((this.satItem.isLinking || this.sat.linkingTrack)) {
    if (occupiedShape) {
      let occupiedLabel = this.satItem.getLabelOfShape(occupiedShape);
      if (occupiedLabel.id === this.id) {
        // if selected a polygon it has, split this polygon out
        if (occupiedShape instanceof Polygon) {
          this.splitShape(occupiedShape);
        }
        this.satItem.resetHiddenMapToDefault();
      } else if (occupiedLabel) {
        // if clicked another label, merge into one
        if (this.polys.length < 1) {
          this.attributes = occupiedLabel.attributes;
          this.categoryPath = occupiedLabel.categoryPath;
        }
        for (let poly of occupiedLabel.polys) {
          this.addShape(poly);
        }
        occupiedLabel.delete();
      }
    }
  } else if (this.state === SegStates.FREE && this.selectedBy(occupiedShape)) {
    this.selectedShape = occupiedShape;
    this.selectedCache = occupiedShape.copy(-1);
    // if clicked on a vertex
    if (occupiedShape instanceof Vertex) {
      if (this.satItem.anyKeyDown(['C', 'D'])) return;
      if (occupiedShape.type === VertexTypes.MIDPOINT) {
        // convert midpoint to a vertex
        this.selectedShape = this.midpointToVertex(occupiedShape);
        // start resize mode
        this.setState(SegStates.RESIZE);
      } else {
        // resize vertex or control points
        this.setState(SegStates.RESIZE);
      }
    } else if (occupiedShape instanceof Polygon && !this.satItem.isLinking
        && !this.sat.linkingTrack && this.satItem.isDown('M')) {
      // if clicked on a polygon, start moving
      this.setState(SegStates.MOVE);
      this.mouseClickPos = mousePos;
      this.bbox = {
        x: this.selectedCache.bbox.min.x,
        y: this.selectedCache.bbox.min.y,
        w: this.selectedCache.bbox.max.x - this.selectedCache.bbox.min.x,
        h: this.selectedCache.bbox.max.y - this.selectedCache.bbox.min.y,
      };
    }
  } else if (this.state === SegStates.DRAW) {
    // draw
  } else if (this.state === SegStates.QUICK_DRAW) {
    // quick draw mode
    let button = document.getElementById('quickdraw_btn');
    if (!this.quickdrawCache.targetPoly && occupiedShape instanceof Polygon) {
      this.quickdrawCache.targetPoly = occupiedShape;
      this.quickdrawCache.targetSeg2d =
          this.satItem.getLabelOfShape(occupiedShape);
      this.quickdrawCache.targetSeg2d.setAsTargeted();
      let shapes = this.quickdrawCache.targetPoly.vertices;
      shapes = shapes.concat(this.newPoly.vertices);
      this.satItem.resetHiddenMap(shapes);
      this.satItem.redrawHiddenCanvas();
      button.innerHTML = 'Select Start Vertex';
    } else if (this.quickdrawCache.targetPoly &&
        !this.quickdrawCache.startVertex &&
        occupiedShape instanceof Vertex &&
        this.quickdrawCache.targetPoly.indexOf(occupiedShape) >= 0) {
      this.quickdrawCache.startVertex = occupiedShape;

      // if occupied object a vertex that is not in polygon, add it
      this.tempPoly.popVertex();

      if (this.tempPoly.vertices.indexOf(occupiedShape) < 0) {
        this.tempPoly.pushVertex(occupiedShape);
        // need below for correct interrupt case
        this.newPoly.pushVertex(occupiedShape);
      }

      this.tempVertex = new Vertex(mousePos.x, mousePos.y,
          VertexTypes.VERTEX, -1);
      this.tempPoly.pushVertex(this.tempVertex);
      this.selectedShape = this.tempVertex;
      button.innerHTML = 'Select End Vertex';
    } else if (this.quickdrawCache.startVertex &&
        !this.quickdrawCache.endVertex &&
        occupiedShape instanceof Vertex &&
        this.quickdrawCache.targetPoly.indexOf(occupiedShape) >= 0 &&
        !this.quickdrawCache.startVertex.equals(occupiedShape)) {
      this.quickdrawCache.endVertex = occupiedShape;

      // if occupied object is a vertex that is not in this polygon, add it
      this.tempPoly.popVertex();
      this.quickdrawCache.shortPathTempPoly = this.tempPoly.copy(-1);
      this.quickdrawCache.longPathTempPoly = this.tempPoly.copy(-1);
      this.quickdrawCache.shortPathTempPoly.pushPath(
          this.quickdrawCache.targetPoly,
          this.quickdrawCache.startVertex,
          this.quickdrawCache.endVertex, false, true);
      this.quickdrawCache.longPathTempPoly.pushPath(
          this.quickdrawCache.targetPoly,
          this.quickdrawCache.startVertex,
          this.quickdrawCache.endVertex, true, true);
      this.tempPoly = this.quickdrawCache.longPath
          ? this.quickdrawCache.longPathTempPoly
          : this.quickdrawCache.shortPathTempPoly;

      // if path is not closed after push path, prepare for draw mode
      if (!occupiedShape.equals(this.newPoly.vertices[0])) {
        this.tempVertex = new Vertex(mousePos.x, mousePos.y,
            VertexTypes.VERTEX, -1);
        this.quickdrawCache.shortPathTempPoly.pushVertex(this.tempVertex);
        this.quickdrawCache.longPathTempPoly.pushVertex(this.tempVertex);
        this.selectedShape = this.tempVertex;
      }

      this.quickdrawCache.targetSeg2d.releaseAsTargeted();
      button.innerHTML = '<kbd>Alt</kbd>   Toggle';
    } else {
      this.endQuickDraw();
    }
  }
};

Seg2d.prototype.doubleclick = function() {
  if (!this.satItem.isLinking && !this.sat.linkingTrack &&
      this.state === SegStates.FREE) {
    let label = this;
    label.satItem.selectLabel(label);
    label.setAsTargeted();
    label.setState(SegStates.FREE);
  }
};

Seg2d.prototype.mouseup = function(e) {
  let mousePos = this.satItem.getMousePos(e);
  if (this.state === SegStates.DRAW) {
    if (!this.tempVertex) {
      this.initDrawing(mousePos);
      return;
    }
    this.tempVertex.xy = [mousePos.x, mousePos.y];
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    if (occupiedShape && (occupiedShape instanceof Vertex)) {
      if (this.newPoly.vertices.indexOf(occupiedShape) === 0) {
        // if occupied object the 1st vertex, close path
        this.tempPoly.popVertex();
        this.completeDrawing();
      } else if (this.newPoly.vertices.indexOf(occupiedShape) < 0) {
        // if occupied object a vertex that is not in polygon, add it
        this.tempPoly.popVertex();
        let pushed = false;
        if (this.latestSharedVertex) {
          for (let label of this.satItem.labels) {
            for (let edge of label.getEdges()) {
              if ((edge.src === occupiedShape
                && edge.dest === this.latestSharedVertex) ||
                (edge.dest === occupiedShape
                  && edge.src === this.latestSharedVertex)) {
                this.newPoly.pushVertex(occupiedShape, edge);
                this.tempPoly.pushVertex(occupiedShape, edge);
                pushed = true;
                break;
              }
            }
            if (pushed) break;
          }
        }

        if (!pushed) {
          this.newPoly.pushVertex(occupiedShape);
          this.tempPoly.pushVertex(occupiedShape);
        }
        this.latestSharedVertex = occupiedShape;

        this.tempVertex = new Vertex(mousePos.x, mousePos.y,
          VertexTypes.VERTEX, -1);
        this.tempPoly.pushVertex(this.tempVertex);
        this.selectedShape = this.tempVertex;
      }
    } else {
      // if occupied object null or not vertex, add tempVertex
      this.tempPoly.popVertex();
      let newVertex = this.tempVertex.copy();
      this.newPoly.pushVertex(newVertex);
      this.tempPoly.pushVertex(newVertex);
      this.satItem.pushToHiddenMap([newVertex]);
      this.latestSharedVertex = null;

      this.tempVertex = new Vertex(mousePos.x, mousePos.y,
        VertexTypes.VERTEX, -1);
      this.tempPoly.pushVertex(this.tempVertex);
      this.selectedShape = this.tempVertex;
    }
  } else if (this.state === SegStates.RESIZE) {
    if (!this.satItem.shapesValid()) {
      this.selectedShape.xy = this.selectedCache.xy;
    }
    this.selectedCache.delete();
    this.selectedCache = null;
    this.setState(SegStates.FREE);
    if (this.parent) {
      this.parent.interpolate(this);
    }
  } else if (this.state === SegStates.FREE && !this.satItem.isLinking &&
      (typeof this.sat.linkingTrack === 'undefined'
          || this.sat.linkingTrack === null)) {
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    let relevant = this.selectedBy(occupiedShape);
    if (relevant && occupiedShape instanceof Vertex) {
      if (occupiedShape.type === VertexTypes.VERTEX &&
        this.satItem.isDown('D')) {
        // deleting a vertex
        for (let label of this.satItem.labels) {
          for (let poly of label.polys) {
            if (poly.vertices.length > 3) {
              let index = poly.vertices.indexOf(this.hoveredShape);
              if (index >= 0) {
                // in vertices, and # vertices > 3, delete
                poly.deleteVertex(index);
              }
            }
          }
        }
      } else if (this.satItem.isDown('C')) {
        if (occupiedShape.type === VertexTypes.MIDPOINT) {
          // convert midpoint to bezier control points
          this.midpointToBezierControl(occupiedShape);
        } else if (occupiedShape.type === VertexTypes.CONTROL_POINT) {
          // convert bezier control points to midpoint
          this.bezierControlToMidpoint(occupiedShape);
        }
      }
      this.setState(SegStates.FREE);
    } else if (!relevant) {
      // deselects self
      this.satItem.deselectAll();
    }
  } else if (this.state === SegStates.MOVE) {
    this.mouseClickPos = null;
    this.bbox = null;
    if (this.parent) {
      this.parent.interpolate(this); // TODO
    }
    this.setState(SegStates.FREE);
  }
};

Seg2d.prototype.mousemove = function(e) {
  let mousePos = this.satItem.getMousePos(e);
  // handling according to state
  if ((this.state === SegStates.DRAW || this.state === SegStates.QUICK_DRAW)
    && !this.satItem.isMouseDown && this.tempVertex) {
    this.tempVertex.xy = [mousePos.x, mousePos.y];
    // hover over vertex
    let hoveredObject = this.satItem.getOccupiedShape(mousePos);
    this.hoveredShape = null;
    let relevant = hoveredObject instanceof Vertex &&
      (this.newPoly.vertices.indexOf(hoveredObject) >= 0);
    if (relevant) {
      this.hoveredShape = hoveredObject;
    }
  } else if (this.state === SegStates.RESIZE
    && this.selectedShape instanceof Vertex) {
    this.selectedShape.xy = [mousePos.x, mousePos.y];
    this.hoveredShape = this.selectedShape;
  } else if (this.state === SegStates.FREE) {
    // hover over vertex
    let hoveredObject = this.satItem.getOccupiedShape(mousePos);
    this.hoveredShape = null;
    for (let poly of this.polys) {
      let relevant = hoveredObject instanceof Vertex &&
        (poly.vertices.indexOf(hoveredObject) >= 0
          || poly.control_points.indexOf(hoveredObject) >= 0);
      if (relevant) {
        this.hoveredShape = hoveredObject;
        break;
      }
    }
  } else if (this.state === SegStates.MOVE) {
    let dx = mousePos.x - this.mouseClickPos.x;
    let dy = mousePos.y - this.mouseClickPos.y;
    // make moved box within the canvas

    dx = Math.min(dx,
      (this.satItem.image.width - this.bbox.w) - this.bbox.x);
    dx = Math.max(dx, -this.bbox.x);
    dy = Math.min(dy,
      (this.satItem.image.height - this.bbox.h) - this.bbox.y);
    dy = Math.max(dy, -this.bbox.y);

    for (let i = 0; i < this.selectedShape.vertices.length; i++) {
      this.selectedShape.vertices[i].x = this.selectedCache.vertices[i].x + dx;
      this.selectedShape.vertices[i].y = this.selectedCache.vertices[i].y + dy;
    }
    for (let i = 0; i < this.selectedShape.control_points.length; i++) {
      this.selectedShape.control_points[i].x =
        this.selectedCache.control_points[i].x + dx;
      this.selectedShape.control_points[i].y =
        this.selectedCache.control_points[i].y + dy;
    }
  }
};

Seg2d.prototype.mouseleave = function (e) { // eslint-disable-line
  if (this.state === SegStates.RESIZE) {
    this.hoveredShape = null;
    this.setState(SegStates.FREE);
  } else if (this.state === SegStates.MOVE) {
    this.mouseClickPos = null;
    this.bbox = null;
    this.setState(SegStates.FREE);
  }
};

/**
 * Key down handler.
 * @param {type} e: Description.
 */
Seg2d.prototype.keydown = function(e) {
  let keyID = e.KeyCode ? e.KeyCode : e.which;
  if (this.satItem.isDown('ctrl')) {
    // key down when ctrl is pressed
    if (keyID === 68) {
      e.preventDefault();
      // ctrl-d for quick draw
      if ($('#quickdraw_btn').length) {
        this.handleQuickdraw();
      }
      this.satItem.ctrlCommandPressed();
    } else if (keyID === 76 &&
        $('#link_btn').length) {
      e.preventDefault();
      // ctrl-l for linking
      this.satItem.linkHandler();
      this.satItem.ctrlCommandPressed();
    } else if (keyID === 46 || keyID === 8) {
      e.preventDefault();
      // ctrl-delete or ctrl-backspace for relabeling a seg2d
      if (this.state === SegStates.FREE) {
        this.polyBuffer = [];
        for (let poly of this.polys) {
          this.polyBuffer.push(poly);
        }
        this.polys = [];
        this.setState(SegStates.DRAW);
        this.satItem.redrawLabelCanvas();
      }
      this.satItem.ctrlCommandPressed();
    }
  } else {
    // solo key down
    if (keyID === 27) { // Esc
      if (this.polyBuffer) {
        for (let poly of this.polyBuffer) {
          this.polys.push(poly);
        }
        this.polyBuffer = null;
      }
      if (this.state === SegStates.QUICK_DRAW) {
        this.endQuickDraw();
      }
      this.setState(SegStates.FREE);
    } else if (keyID === 18 && this.state === SegStates.QUICK_DRAW) {
      // alt toggle long path mode in quick draw
      this.quickdrawCache.longPath = !this.quickdrawCache.longPath;
      if (this.quickdrawCache.shortPathTempPoly
          && this.quickdrawCache.longPathTempPoly) {
        this.tempPoly = this.quickdrawCache.longPath
            ? this.quickdrawCache.longPathTempPoly
            : this.quickdrawCache.shortPathTempPoly;
      }
    } else if (keyID === 13) {
      // enter for ending a Path object
      if (this.state === SegStates.DRAW && !Seg2d.closed) {
        this.newPoly.endPath();

        if (this.newPoly.isValidShape()) {
          this.addShape(this.newPoly);
          this.tempVertex = null;
          this.tempPoly = null;
        }

        this.setState(SegStates.FREE);
        this.tempVertex = null;
        this.selectedShape = this.newPoly;
      }
      // end linking
      if (this.satItem.isLinking) {
        this.satItem.linkHandler();
      }
    } else if (keyID === 68) {
      // d for deleting the last labeled vertex while drawing
      if (this.state === SegStates.DRAW) {
        if (this.newPoly.vertices.length < 2) {
          // set state to free to be deleted
          this.setState(SegStates.FREE);
        } else {
          // otherwise, pop the last labeled vertex
          this.newPoly.popVertex();
          this.tempPoly.popVertex();
          this.tempPoly.popVertex();
          this.tempPoly.pushVertex(this.tempVertex);
        }
      }
    }
  }
};
