
/* global ImageLabel SatImage Shape Polygon Path Vertex */
/* global rgba VertexTypes EdgeTypes*/
/* global GRAYOUT_COLOR SELECT_COLOR LINE_WIDTH OUTLINE_WIDTH
ALPHA_HIGH_FILL ALPHA_LOW_FILL ALPHA_LINE UP_RES_RATIO FONT_SIZE */
/* exported Seg2d*/

// constants
let SegStates = Object.freeze({
  FREE: 0, DRAW: 1, RESIZE: 2, QUICK_DRAW: 3, LINK: 4});

/**
 * 2D segmentation label
 * @param {Sat} sat: context
 * @param {int} id: label id
 * @param {object} optionalAttributes - Optional attributes
 */
function Seg2d(sat, id, optionalAttributes) {
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
    if (Seg2d.closed) {
      this.newPoly = new Polygon();
      this.tempPoly = new Polygon();
    } else {
      this.newPoly = new Path();
      this.tempPoly = new Path();
    }
    this.setState(SegStates.DRAW);

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

    this.tempVertex = new Vertex(mousePos.x, mousePos.y);
    this.tempPoly.pushVertex(this.tempVertex);
    this.selectedShape = null;
  }
}

Seg2d.prototype = Object.create(ImageLabel.prototype);

Seg2d.useCrossHair = false;
Seg2d.defaultCursorStyle = 'auto';
Seg2d.useDoubleClick = true;
Seg2d.closed =
    document.getElementById('label_type').innerHTML === 'segmentation';

Seg2d.prototype.identityInterpolation = function(startLabel, endLabel, weight) { // eslint-disable-line
  this.deleteAllPolyline();
  let targetLabel;
  if (startLabel) {
    targetLabel = startLabel;
  } else {
    targetLabel = endLabel;
  }
  for (let poly of targetLabel.polys) {
    this.addPolyline(poly.copy());
  }
  this.setState(SegStates.FREE);
};

Seg2d.prototype.addPolyline = function(poly) {
  this.polys = this.polys.concat(poly);
};

/**
 * Split a given polyline from this Seg2d label,
 * and create a new Seg2d label for it.
 * @param {Polyline} poly - the polyline
 */
Seg2d.prototype.splitPolyline = function(poly) {
  for (let i = 0; i < this.polys.length; i++) {
    if (this.polys[i] === poly) {
      this.polys.splice(i, 1);
      // create a Seg2d label for this polygon
      let label = this.sat.newLabel({
        categoryPath: this.categoryPath, attributes: this.attributes,
        mousePos: null,
      });
      label.addPolyline(poly);
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
    this.polys.splice(i, 1);
  }
};

Seg2d.prototype.setAsTargeted = function() {
  this.targeted = true;
  if (this.satItem.active && this.state === SegStates.FREE) {
    this.satItem.resetHiddenMap(this.getAllHiddenShapes());
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
      let quickdrawButton = document.getElementById('quickdraw_btn');
      quickdrawButton.innerHTML = '<kbd>s</kbd> Quickdraw';
      quickdrawButton.style.backgroundColor = 'white';
      let linkButton = document.getElementById('link_btn');
      linkButton.innerHTML = 'Link';
      linkButton.style.backgroundColor = 'white';

      // clean up cache
      if (this.quickdrawCache.targetSeg2d) {
        this.quickdrawCache.targetSeg2d.releaseAsTargeted();
      }
    }

    this.state = state;
    this.selectedShape = null;

    // reset hiddenMap with this polygon and corresponding handles
    if (this.satItem.active) {
      this.satItem.resetHiddenMap(this.getAllHiddenShapes());
      this.satItem.redrawHiddenCanvas();
    }
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
  } else if (state === SegStates.LINK) {
    this.state = state;
    this.satItem.resetHiddenMapToDefault();
    this.satItem.redrawHiddenCanvas();
  } else if (state === SegStates.QUICK_DRAW) {
    this.state = state;
    if (Seg2d.closed) {
      this.quickdrawCache = {};
    }
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
  }
};

/**
 * Function to set the tool box of Seg2d.
 * @param {object} satItem - the SatItem object.
 */
Seg2d.setToolBox = function(satItem) {
  if (Seg2d.closed) {
    satItem.isLinking = false;
    document.getElementById('link_btn').onclick = function()
    {satItem._linkHandler();};
    document.getElementById('quickdraw_btn').onclick = function() {
      if (satItem.selectedLabel) {
        satItem.selectedLabel.handleQuickdraw();
      }
    };
  }
};

/**
 * Convert midpoint to a new vertex, do nothing if not a midpoint
 * @param {object} pt: the position of the midpoint as vertex object
 */
Seg2d.prototype.midpointToVertex = function(pt) {
  let edge1 = null;
  let edge2 = null;
  for (let label of this.satItem.labels) {
    for (let poly of label.polys) {
      let _edge1 = null;
      let _edge2 = null;
      poly.alignEdges();
      [_edge1, _edge2] = poly.midpointToVertex(pt, edge1, edge2);
      if (!_edge1 && !_edge2) {
        poly.reverse();
        [_edge1, _edge2] = poly.midpointToVertex(pt, edge1, edge2);
      }
      if (!edge1 && !edge2) {
        [edge1, edge2] = [_edge1, _edge2];
      }
    }
  }
};

/**
 * Link button handler
 */
SatImage.prototype._linkHandler = function() {
  let button = document.getElementById('link_btn');
  if (!this.isLinking) {
    this.isLinking = true;
    button.innerHTML = 'Finish Linking';
    button.style.backgroundColor = 'lightgreen';

    let cat = this.catSel.options[this.catSel.selectedIndex].innerHTML;
    if (!this.selectedLabel) {
      let attributes = {};
      for (let i = 0; i < this.sat.attributes.length; i++) {
        if (this.sat.attributes[i].toolType === 'switch') {
          attributes[this.sat.attributes[i].name] = false;
        } else if (this.sat.attributes[i].toolType === 'list') {
          attributes[this.sat.attributes[i].name] = [0,
            this.sat.attributes[i].values[0]];
        }
      }
      this.selectedLabel = this.sat.newLabel({
        categoryPath: cat, occl: false,
        trunc: false, mousePos: null,
      });
      this.selectedLabel.setAsTargeted();
    }
  } else {
    this.isLinking = false;
    button.innerHTML = 'Link';
    button.style.backgroundColor = 'white';
    this.updateLabelCount();
  }
  this.selectedLabel.linkHandler();
};

Seg2d.prototype.linkHandler = function() {
  if (this.state === SegStates.FREE) {
    this.setState(SegStates.LINK);
  } else if (this.state === SegStates.LINK) {
    this.setState(SegStates.FREE);
    if (this.polys.length < 1) {
      this.delete();
    }
  }
};

Seg2d.prototype.handleQuickdraw = function() {
  let button = document.getElementById('quickdraw_btn');
  if (this.state === SegStates.DRAW) {
    // s switch to quick draw
    button.innerHTML = 'Select Target Polygon';
    button.style.backgroundColor = 'lightgreen';
    this.setState(SegStates.QUICK_DRAW);
  } else if (this.state === SegStates.QUICK_DRAW) {
    button.innerHTML = '<kbd>s</kbd> Quickdraw';
    button.style.backgroundColor = 'white';
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
        this.addPolyline(this.newPoly);
        this.tempVertex.delete();
        this.tempPoly.delete();
        this.tempVertex = null;
        this.tempPoly = null;
      }

      this.setState(SegStates.FREE);
      this.selectedShape = this.newPoly;
    } else {
      this.setState(SegStates.DRAW);
    }
  }
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
        this.addPolyline(Polygon.fromJson(polyJson));
      } else {
        this.addPolyline(Path.fromJson(polyJson));
      }
    }
  }
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
  if (this.polys.length < 1) {
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
Seg2d.prototype.redrawMainCanvas = function(mainCtx) {
  // set context font
  mainCtx.font = FONT_SIZE * UP_RES_RATIO + 'px Verdana';
  mainCtx.save(); // save the canvas context settings
  let styleColor = this.styleColor();
  mainCtx.strokeStyle = styleColor;
  this.setPolygonLine(mainCtx);
  this.setPolygonFill(mainCtx);

  if (this.state === SegStates.DRAW || this.state === SegStates.QUICK_DRAW) {
    this.tempPoly.draw(mainCtx, this.satItem,
      this.isTargeted() && this.state !== SegStates.DRAW
      && this.state !== SegStates.QUICK_DRAW);
    this.tempPoly.drawHandles(mainCtx, this.satItem, styleColor, null, false);
  } else if (this.polys.length > 0) {
    for (let poly of this.polys) {
      poly.draw(mainCtx, this.satItem,
        this.isTargeted() && this.state !== SegStates.DRAW
        && this.state !== SegStates.QUICK_DRAW);
      if (this.isTargeted()) {
        poly.drawHandles(mainCtx,
          this.satItem, styleColor, this.hoveredShape, true);
      }
    }
    mainCtx.fillStyle = this.styleColor();
    this.drawTag(mainCtx, this.polys[0].centroidCoords());
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
  if (shape) {
    return this.defaultCursorStyle;
  }
  return this.defaultCursorStyle;
};

Seg2d.prototype.mousedown = function(e) {
  let mousePos = this.satItem.getMousePos(e);

  let occupiedShape = this.satItem.getOccupiedShape(mousePos);
  if (this.state === SegStates.FREE && occupiedShape) {
    // if clicked on midpoint, convert to vertex
    if (occupiedShape instanceof Vertex) {
      if (occupiedShape.type === VertexTypes.MIDPOINT) {
        Shape.registerShape(occupiedShape); // assign id to new vertex
        this.midpointToVertex(occupiedShape);
      }

      // start resize mode
      if (occupiedShape instanceof Vertex) {
        this.setState(SegStates.RESIZE);
        this.selectedShape = occupiedShape;
        this.selectedCache = occupiedShape.copy();
      }
    }
  } else if (this.state === SegStates.DRAW) {
    // draw
  } else if (this.state === SegStates.QUICK_DRAW) {
    // quick draw mode
    let button = document.getElementById('quickdraw_btn');
    if (!occupiedShape) {
      // TODO: if nothing is clicked, return to DRAW state
      if (this.quickdrawCache.targetPoly) { // must be before state transition
        this.quickdrawCache.targetSeg2d.releaseAsTargeted();
      }
      if (this.newPoly.vertices.length > 1
        && this.quickdrawCache.endVertex
        && this.quickdrawCache.endVertex.equals(this.newPoly.vertices[0])) {
        // if occupied object the 1st vertex, close path
        this.tempPoly.popVertex();
        this.newPoly.endPath();

        if (this.newPoly.isValidShape()) {
          this.addPolyline(this.newPoly);
          this.tempVertex.delete();
          this.tempPoly.delete();
          this.tempVertex = null;
          this.tempPoly = null;
        }

        this.setState(SegStates.FREE);
        this.selectedShape = this.newPoly;
      } else {
        this.setState(SegStates.DRAW);
        // change back the indicator
        button.innerHTML = '<kbd>s</kbd> Quickdraw';
        button.style.backgroundColor = 'white';
      }
    } else if (this.newPoly.vertices.length > 1
      && occupiedShape.id === this.newPoly.vertices[0].id
      && !this.quickdrawCache.endVertex
      && this.quickdrawCache.targetPoly
      && this.quickdrawCache.targetPoly.indexOf(occupiedShape) < 0) {
      // if occupied object the 1st vertex, change to draw mode to close path
      this.setState(SegStates.DRAW);
    } else if (!this.quickdrawCache.targetPoly) {
      if (occupiedShape instanceof Polygon
        && !this.newPoly.equals(occupiedShape)) {
        this.quickdrawCache.targetPoly = occupiedShape;
        this.quickdrawCache.targetSeg2d =
          this.satItem.getLabelOfShape(occupiedShape);
        this.quickdrawCache.targetSeg2d.setAsTargeted();
        let shapes = this.quickdrawCache.targetPoly.vertices;
        shapes = shapes.concat(this.newPoly.vertices);
        this.satItem.resetHiddenMap(shapes);
        this.satItem.redrawHiddenCanvas();
        button.innerHTML = 'Select Start Vertex';
      }
    } else if (!this.quickdrawCache.startVertex) {
      if (occupiedShape instanceof Vertex
        && occupiedShape.type === VertexTypes.VERTEX
        && this.quickdrawCache.targetPoly.indexOf(occupiedShape) >= 0) {
        this.quickdrawCache.startVertex = occupiedShape;

        // if occupied object a vertex that is not in polygon, add it
        this.tempPoly.popVertex();

        if (this.tempPoly.vertices.indexOf(occupiedShape) < 0) {
          this.tempPoly.pushVertex(occupiedShape);
          // need below for correct interrupt case
          this.newPoly.pushVertex(occupiedShape);
        }

        this.tempVertex = new Vertex(mousePos.x, mousePos.y);
        this.tempPoly.pushVertex(this.tempVertex);
        this.selectedShape = this.tempVertex;
        button.innerHTML = 'Select End Vertex';
      }
    } else if (!this.quickdrawCache.endVertex) {
      if (occupiedShape instanceof Vertex
        && occupiedShape.type === VertexTypes.VERTEX
        && this.quickdrawCache.targetPoly.indexOf(occupiedShape) >= 0
        && !this.quickdrawCache.startVertex.equals(occupiedShape)) {
        this.quickdrawCache.endVertex = occupiedShape;

        // if occupied object is a vertex that is not in this polygon, add it
        this.tempPoly.popVertex();
        // this.tempPoly.popVertex();
        this.quickdrawCache.shortPathTempPoly = this.tempPoly.copy(true);
        this.quickdrawCache.longPathTempPoly = this.tempPoly.copy(true);
        this.quickdrawCache.shortPathTempPoly.pushPath(
          this.quickdrawCache.targetPoly,
          this.quickdrawCache.startVertex,
          this.quickdrawCache.endVertex);
        this.quickdrawCache.longPathTempPoly.pushPath(
          this.quickdrawCache.targetPoly,
          this.quickdrawCache.startVertex,
          this.quickdrawCache.endVertex, true);
        this.quickdrawCache.shortPathPoly =
          this.quickdrawCache.shortPathTempPoly.copy(true);
        this.quickdrawCache.longPathPoly =
          this.quickdrawCache.longPathTempPoly.copy(true);
        this.tempPoly = this.quickdrawCache.longPath
          ? this.quickdrawCache.longPathTempPoly
          : this.quickdrawCache.shortPathTempPoly;
        this.newPoly = this.quickdrawCache.longPath
          ? this.quickdrawCache.longPathPoly
          : this.quickdrawCache.shortPathPoly;

        // if path is not closed after push path, prepare for draw mode
        if (!occupiedShape.equals(this.newPoly.vertices[0])) {
          this.tempVertex = new Vertex(mousePos.x, mousePos.y);
          this.quickdrawCache.shortPathTempPoly.pushVertex(this.tempVertex);
          this.quickdrawCache.longPathTempPoly.pushVertex(this.tempVertex);
          this.selectedShape = this.tempVertex;
        }

        this.quickdrawCache.targetSeg2d.releaseAsTargeted();
        button.innerHTML = '<kbd>s</kbd> End Quickdraw\n' +
          '<kbd>Alt</kbd> Toggle path';
      }
    } else if (occupiedShape.id === this.newPoly.vertices[0].id) {
      // user want to close the polygon after path is pushed
      // note: must toggle path before this action!!
      this.setState(SegStates.DRAW);
    }
  } else if (this.state === SegStates.LINK && occupiedShape) {
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    let occupiedLabel = this.satItem.getLabelOfShape(occupiedShape);
    if (occupiedLabel.id === this.id) {
      // if selected a polygon it has, split this polygon out
      if (occupiedShape instanceof Polygon) {
        this.splitPolyline(occupiedShape);
      }
      this.satItem.resetHiddenMapToDefault();
    } else if (occupiedLabel) {
      // if clicked another label, merge into one
      if (this.polys.length < 1) {
        this.attributes = occupiedLabel.attributes;
        this.categoryPath = occupiedLabel.categoryPath;
      }
      for (let poly of occupiedLabel.polys) {
        this.addPolyline(poly);
      }
      occupiedLabel.delete();
    }
  }
};

Seg2d.prototype.doubleclick = function() {
  if (!this.satItem.isLinking && this.state === SegStates.FREE) {
    let label = this;
    label.satItem.selectLabel(label);
    label.setAsTargeted();
    label.setState(SegStates.FREE);
  }
};

Seg2d.prototype.mouseup = function(e) {
  let mousePos = this.satItem.getMousePos(e);
  if (this.state === SegStates.DRAW) {
    this.tempVertex.xy = [mousePos.x, mousePos.y];
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    if (occupiedShape && (occupiedShape instanceof Vertex)) {
      if (this.newPoly.vertices.indexOf(occupiedShape) === 0) {
        // if occupied object the 1st vertex, close path
        this.tempPoly.popVertex();

        if (this.latestSharedVertex) {
          let pushed = false;
          let firstVertex = this.newPoly.vertices[0];
          for (let label of this.satItem.labels) {
            for (let edge of label.getEdges()) {
              if ((edge.src === firstVertex
                && edge.dest === this.latestSharedVertex) ||
                (edge.dest === firstVertex
                  && edge.src === this.latestSharedVertex)) {
                this.newPoly.edges.pop();
                this.newPoly.edges.push(edge);
                pushed = true;
                break;
              }
            }
            if (pushed) break;
          }
        }
        this.newPoly.endPath();

        if (this.newPoly.isValidShape()) {
          this.addPolyline(this.newPoly);
          this.tempVertex.delete();
          this.tempPoly.delete();
          this.tempVertex = null;
          this.tempPoly = null;
        }

        this.setState(SegStates.FREE);
        this.selectedShape = this.newPoly;
        if (this.parent) {
          this.parent.interpolate(this);
        }
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

        this.tempVertex = new Vertex(mousePos.x, mousePos.y);
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

      this.tempVertex = new Vertex(mousePos.x, mousePos.y);
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
  } else if (this.state === SegStates.FREE) {
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    if (!occupiedShape) {
      // deselects self
      this.satItem.deselectAll();
    }
  }
};

Seg2d.prototype.mousemove = function(e) {
  let mousePos = this.satItem.getMousePos(e);
  // handling according to state
  if ((this.state === SegStates.DRAW || this.state === SegStates.QUICK_DRAW)
    && !this.satItem.isMouseDown) {
    this.tempVertex.xy = [mousePos.x, mousePos.y];
  } else if (this.state === SegStates.RESIZE
    && this.selectedShape instanceof Vertex) {
    this.selectedShape.xy = [mousePos.x, mousePos.y];
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
      } else {
        this.hoveredShape = null;
      }
    }
  }
};

Seg2d.prototype.mouseleave = function(e) { // eslint-disable-line
  if (this.state === SegStates.RESIZE) {
    this.hoveredShape = null;
    this.setState(SegStates.FREE);
  }
};

/**
 * Key down handler.
 * @param {type} e: Description.
 */
Seg2d.prototype.keydown = function(e) {
  let keyID = e.KeyCode ? e.KeyCode : e.which;
  if (keyID === 27) { // Esc
    this.setState(SegStates.FREE);
  } else if (keyID === 85) {
    // u for unlinking the selected label
    if (this.polys.length > 1) {
      for (let poly of this.polys) {
        this.splitPolyline(poly);
      }
    }
  } else if (keyID === 66) {
    // b for showing bezier control points
    if (this.state === SegStates.FREE &&
      this.hoveredShape &&
      this.hoveredShape.type === VertexTypes.MIDPOINT) {
      for (let poly of this.polys) {
        for (let edge of poly.edges) {
          if (edge.control_points[0] === this.hoveredShape) {
            edge.type = EdgeTypes.BEZIER;
            this.setState(SegStates.FREE);
            return;
          }
        }
      }
    }
  } else if (keyID === 83) {
    this.handleQuickdraw();
  } else if (keyID === 18 && this.state === SegStates.QUICK_DRAW) {
    // alt toggle long path mode in quick draw
    this.quickdrawCache.longPath = !this.quickdrawCache.longPath;
    if (this.quickdrawCache.shortPathTempPoly
      && this.quickdrawCache.longPathTempPoly) {
      this.tempPoly = this.quickdrawCache.longPath
        ? this.quickdrawCache.longPathTempPoly
        : this.quickdrawCache.shortPathTempPoly;
      this.newPoly = this.quickdrawCache.longPath
        ? this.quickdrawCache.longPathPoly
        : this.quickdrawCache.shortPathPoly;
    }
  } else if (keyID === 13 && !Seg2d.closed) {
    // enter for ending a Path object
    if (this.state === SegStates.DRAW) {
      this.newPoly.endPath();

      if (this.newPoly.isValidShape()) {
        this.addPolyline(this.newPoly);
        this.tempVertex = null;
        this.tempPoly = null;
      }

      this.setState(SegStates.FREE);
      this.tempVertex = null;
      this.selectedShape = this.newPoly;
    }
  }
};
