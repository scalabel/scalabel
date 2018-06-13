
/* global ImageLabel SatImage Polygon Vertex */
/* global rgba VertexTypes EdgeTypes*/
/* global GRAYOUT_COLOR SELECT_COLOR LINE_WIDTH OUTLINE_WIDTH
ALPHA_HIGH_FILL ALPHA_LOW_FILL ALPHA_LINE */
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

  let mousePos;
  if (optionalAttributes) {
    mousePos = optionalAttributes.mousePos;
    this.categoryPath = optionalAttributes.categoryPath;
    for (let i = 0; i < this.sat.attributes.length; i++) {
      let attributeName = this.sat.attributes[i].name;
      if (attributeName in optionalAttributes.attributes) {
        this.attributes[attributeName] =
          optionalAttributes.attributes[attributeName];
      }
    }
  }

  if (mousePos) {
    // if mousePos given, start drawing
    this.setState(SegStates.DRAW);
    this.newPoly = new Polygon();
    this.tempPoly = new Polygon();

    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    if (occupiedShape instanceof Vertex) {
      this.newPoly.pushVertex(occupiedShape);
      this.tempPoly.pushVertex(occupiedShape);
      this.satItem.pushToHiddenMap([occupiedShape]);
    } else {
      let firstVertex = new Vertex(mousePos.x, mousePos.y);
      this.newPoly.pushVertex(firstVertex);
      this.tempPoly.pushVertex(firstVertex);
      this.satItem.pushToHiddenMap([firstVertex]);
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

Seg2d.prototype.addPolygon = function(poly) {
  this.polys = this.polys.concat(poly);
};

/**
 * Split a given polygon from this Seg2d label,
 * and create a new Seg2d label for it.
 * @param {Polygon} poly - the polygon
 */
Seg2d.prototype.splitPolygon = function(poly) {
  for (let i = 0; i < this.polys.length; i++) {
    if (this.polys[i] === poly) {
      this.polys.splice(i, 1);
      // create a Seg2d label for this polygon
      let label = this.sat.newLabel({
        categoryPath: this.categoryPath, attributes: this.attributes,
        mousePos: null,
      });
      label.addPolygon(poly);
      label.releaseAsTargeted();
      break;
    }
  }
};

/**
 * Delete a given polygon from this Seg2d label.
 * @param {Polygon} poly - the polygon
 */
Seg2d.prototype.deletePolygon = function(poly) {
  for (let i = 0; i < this.polys.length; i++) {
    if (this.polys[i] === poly) {
      this.polys[i].delete();
      this.polys.splice(i, 1);
      break;
    }
  }
};

/**
 * Function to set the current state.
 * @param {number} state - The state to set to.
 */
Seg2d.prototype.setState = function(state) {
  if (state === SegStates.FREE) {
    this.state = state;
    this.selectedShape = null;

    // reset hiddenMap with this polygon and corresponding handles
    this.satItem.resetHiddenMap(this.getAllHiddenShapes());
    this.satItem.redrawHiddenCanvas();
  } else if (state === SegStates.DRAW) {
    // reset hiddenMap with all existing vertices
    let shapes = [];
    for (let label of this.satItem.labels) {
      shapes = shapes.concat(label.getVertices());
      shapes = shapes.concat(label.getControlPoints());
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
    this.quickdrawCache = {};
    let shapes = [];
    for (let label of this.satItem.labels) {
      shapes = shapes.concat(label.getPolygons());
    }
    this.satItem.pushToHiddenMap(shapes);
    this.satItem.redrawHiddenCanvas();
  }
};

/**
 * Function to set the tool box of Seg2d.
 * @param {object} satItem - the SatItem object.
 */
Seg2d.setToolBox = function(satItem) {
  satItem.isLinking = false;
  document.getElementById('link_btn').onclick = function()
      {satItem._linkHandler();};
};

/**
 * Link button handler
 */
SatImage.prototype._linkHandler = function() {
  if (!this.isLinking) {
    this.isLinking = true;
    document.getElementById('link_btn').innerHTML = 'Finish Linking';

    let cat = this.catSel.options[this.catSel.selectedIndex].innerHTML;
    if (!this.selectedLabel) {
      let attributes = {};
      for (let i = 0; i < self.sat.attributes.length; i++) {
        if (self.sat.attributes[i].toolType === 'switch') {
          attributes[self.sat.attributes[i].name] = false;
        } else if (self.sat.attributes[i].toolType === 'list') {
          attributes[self.sat.attributes[i].name] = [0,
            self.sat.attributes[i].values[0]];
        }
      }
      this.selectedLabel = this.sat.newLabel({
        categoryPath: cat, occl: false,
        trunc: false, mousePos: null,
      });
    }
  } else {
    this.isLinking = false;
    document.getElementById('link_btn').innerHTML = 'Link';
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

/**
 * Load label data from a encoded string
 * @param {object} data: json representation of label data.
 */
Seg2d.prototype.decodeLabelData = function(data) {
  this.id = data.id;
  this.polys = [];
  if (data.polys) {
    for (let polyJson of data.polys) {
      this.addPolygon(Polygon.fromJson(polyJson));
    }
  }
};

/**
 * Encode the label data into a json object.
 * @return {object} - the encoded json object.
 */
Seg2d.prototype.encodeLabelData = function() {
  let polysJson = [];
  for (let poly of this.polys) {
    polysJson = polysJson.concat(poly.toJson());
  }
  return {
    id: this.id,
    polys: polysJson,
  };
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
  return shapes;
};

/**
 * Draw this bounding box on the canvas.
 * @param {object} mainCtx - HTML canvas context for visible objects.
 */
Seg2d.prototype.redrawMainCanvas = function(mainCtx) {
  // set context font
  mainCtx.font = '13px Verdana';

  mainCtx.save(); // save the canvas context settings
  let styleColor = this.styleColor();
  mainCtx.strokeStyle = styleColor;
  this.setPolygonLine(mainCtx);
  this.setPolygonFill(mainCtx);

  if (this.state === SegStates.DRAW || this.state === SegStates.QUICK_DRAW) {
    this.tempPoly.draw(mainCtx, this.satItem,
        this.isTargeted() && this.state !== SegStates.DRAW
      && this.state !== SegStates.QUICK_DRAW);
    this.tempPoly.drawHandles(mainCtx, this.satItem, styleColor, null);
  } else if (this.polys.length > 0) {
    for (let poly of this.polys) {
      poly.draw(mainCtx, this.satItem,
          this.isTargeted() && this.state !== SegStates.DRAW
        && this.state !== SegStates.QUICK_DRAW);
      if (this.isTargeted()) {
        poly.drawHandles(mainCtx,
            this.satItem, styleColor, this.hoveredShape);
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
    if (this.isValid()) {
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
 * Get whether this Seg2d object is valid.
 * @return {boolean} - True if the Seg2d object is valid.
 */
Seg2d.prototype.isValid = function() {
  if (this.polys.length < 1) {
    return false;
  }
  for (let poly of this.polys) {
    if (!poly.isValid()) {
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
      for (let poly of this.polys) {
        if (poly.control_points.indexOf(occupiedShape) >= 0) {
          poly.midpointToVertex(occupiedShape);
        }
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

        if (this.newPoly.isValid()) {
          this.addPolygon(this.newPoly);
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
    } else if (this.newPoly.vertices.length > 1
      && occupiedShape.id === this.newPoly.vertices[0].id
      && !this.quickdrawCache.startVertex) {
      // if occupied object the 1st vertex, close path
      this.tempPoly.popVertex();
      this.newPoly.endPath();

      if (this.newPoly.isValid()) {
        this.addPolygon(this.newPoly);
        this.tempVertex.delete();
        this.tempPoly.delete();
        this.tempVertex = null;
        this.tempPoly = null;
      }

      this.setState(SegStates.FREE);
      this.selectedShape = this.newPoly;
      this.setAsTargeted();
    } else if (!this.quickdrawCache.targetPoly) {
      if (occupiedShape instanceof Polygon
        && !this.newPoly.equals(occupiedShape)) {
        this.quickdrawCache.targetPoly = occupiedShape;
        this.quickdrawCache.targetSeg2d =
          this.satItem.getLabelOfShape(occupiedShape);
        this.quickdrawCache.targetSeg2d.setAsTargeted();
        this.satItem.pushToHiddenMap(this.quickdrawCache.targetPoly.vertices);
        this.satItem.redrawHiddenCanvas();
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
      }
    }
  } else if (this.state === SegStates.LINK && occupiedShape) {
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    let occupiedLabel = this.satItem.getLabelOfShape(occupiedShape);
    if (occupiedLabel.id === this.id) {
      // if selected a polygon it has, split this polygon out
      if (occupiedShape instanceof Polygon) {
        this.splitPolygon(occupiedShape);
      }
     this.satItem.resetHiddenMapToDefault();
    } else if (occupiedLabel) {
      // if clicked another label, merge into one
      if (this.polys.length < 1) {
        this.attributes = occupiedLabel.attributes;
        this.categoryPath = occupiedLabel.categoryPath;
      }
      for (let poly of occupiedLabel.polys) {
        this.addPolygon(poly);
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
        this.newPoly.endPath();

        if (this.newPoly.isValid()) {
          this.addPolygon(this.newPoly);
          this.tempVertex.delete();
          this.tempPoly.delete();
          this.tempVertex = null;
          this.tempPoly = null;
        }

        this.setState(SegStates.FREE);
        this.selectedShape = this.newPoly;
      } else if (this.newPoly.vertices.indexOf(occupiedShape) < 0) {
        // if occupied object a vertex that is not in polygon, add it
        this.tempPoly.popVertex();
        this.newPoly.pushVertex(occupiedShape);
        this.tempPoly.pushVertex(occupiedShape);

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

      this.tempVertex = new Vertex(mousePos.x, mousePos.y);
      this.tempPoly.pushVertex(this.tempVertex);
      this.selectedShape = this.tempVertex;
    }
  } else if (this.state === SegStates.RESIZE) {
    if (!this.satItem.isValid()) {
      this.selectedShape.xy = this.selectedCache.xy;
    }
    this.selectedCache.delete();
    this.selectedCache = null;
    this.setState(SegStates.FREE);
  } else if (this.state === SegStates.FREE) {
    let occupiedShape = this.satItem.getOccupiedShape(mousePos);
    if (!occupiedShape) {
      // deselects self
      this.releaseAsTargeted();
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

/**
 * Key down handler.
 * @param {type} e: Description.
 */
Seg2d.prototype.keydown = function(e) {
  let keyID = e.KeyCode ? e.KeyCode : e.which;
  if (keyID === 27) { // Esc
    // if not completed, delete this label
    if (this.state = SegStates.DRAW) {
      this.delete();
      this.satItem.deselectAll();
    }
  } else if (keyID === 85) {
    // u for unlinking the selected label
    if (this.polys.length > 1) {
      for (let poly of this.polys) {
        this.splitPolygon(poly);
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
    if (this.state === SegStates.DRAW) {
      // s switch to quick draw
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

        if (this.newPoly.isValid()) {
          this.addPolygon(this.newPoly);
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
  }
};
