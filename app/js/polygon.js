/* global module */

// Define Enums
const VertexTypes = {VERTEX: 'vertex', MIDPOINT: 'midpoint',
  CONTROL_POINT: 'control_point'};
const EdgeTypes = {LINE: 'line', BEZIER: 'bezier'};

// reduce functions for vertices
// the vertex with min x and y
const minVertex = function(p, c) {
  return new Vertex(Math.min(p.x, c.x), Math.min(p.y, c.y));
};

// the vertex with max x and y
const maxVertex = function(p, c) {
  return new Vertex(Math.max(p.x, c.x), Math.max(p.y, c.y));
};

const meanVertex = function(p, c, i, a) {
  return new Vertex(p.x+(c.x/a.length), p.y+(c.y/a.length));
};

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
  module.exports = [Vertex, Edge, Polygon, VertexTypes, EdgeTypes];
}

/**
 * Shape class
 * base class for all shapes, holds necessary
 * attributes needed and managed by Sat
 */
function Shape() {
  this.index = null;
}

/**
 * Vertex Class is designed to be non-traversable by itself,
 * because no vertex will be selected without a known
 * polygon that contains the vertex, so polygon object is responsible
 * for all the traversal needs. This design keeps a clean cut in
 * object responsibilities and avoids heterogeneous circular
 * references.
 * P.S. everything is in image coordinate
 * @param {int} x: x-coordinate
 * @param {int} y: y-coordinate
 * @param {object} type: type of the vertex
 */
function Vertex(x=0, y=0, type=VertexTypes.VERTEX) {
  Shape.call(this);
  this._x = x;
  this._y = y;
  this.type = type;

  // constants
  this.CONTROL_FILL_COLOR = [255, 255, 255];
  this.CONTROL_LINE_COLOR = [0, 0, 0];
}

Vertex.prototype = Object.create(Shape.prototype);

Object.defineProperty(Vertex.prototype, 'x', {
  get: function() {return this._x;},
  set: function(x) {this._x = x;},
});

Object.defineProperty(Vertex.prototype, 'y', {
  get: function() {return this._y;},
  set: function(y) {this._y = y;},
});

Object.defineProperty(Vertex.prototype, 'xy', {
  get: function() {return [this._x, this._y];},
  set: function(xy) {
    let [x, y] = xy;
    this.x = x;
    this.y = y;
  },
});

Object.defineProperty(Vertex.prototype, 'x_int', {
  get: function() {return Math.round(this.x);},
});

Object.defineProperty(Vertex.prototype, 'y_int', {
  get: function() {return Math.round(this.y);},
});

/**
 * Interpolation toward a target point
 * @param {object} v: target vertex
 * @param {float} f: fraction to interpolate towards v
 * @return {object} interpolated point
 */
Vertex.prototype.interpolate = function(v, f) {
  return new Vertex( this.x + (v.x - this.x) * f, this.y + (v.y - this.y) * f );
};

/**
 * Calculate distance from the current vertex to another
 * @param {object} v: target vertex
 * @return {int} distance
 */
Vertex.prototype.distanceTo = function(v) {
  let a = this.x - v.x;
  let b = this.y - v.y;
  return Math.sqrt( a * a + b * b );
};

Vertex.prototype.toJson = function() {
  return {x: this.x, y: this.y};
};

Vertex.fromJson = function(json) {
  return new Vertex( json.x, json.y );
};

// Deep Copy
Vertex.prototype.copy = function() {
  return new Vertex( this.x, this.y );
};

Vertex.prototype.equals = function(v, threshold=1e-6) {
  return Math.abs(this.x - v.x) < threshold
    && Math.abs(this.y - v.y) < threshold;
};

/**
 * Edge Class, take in objects as input to keep references clean
 * Principle: redundant information is not stored as attributes
 * but written as getters instead to keep serialization clean.
 * Note: setting vertex reference re-initializes control points
 * including for bezier. To drag a bezier control point (pt) to
 * position [x, y], simply call pt.xy = [x, y]
 * @param {object} src: source vertex
 * @param {object} dest: destination vertex
 * @param {string} type: connection type, line or bezier
 */
function Edge(src=new Vertex(), dest=new Vertex(), type=EdgeTypes.LINE) {
  Shape.call(this);
  this._src = src; // define private variables to avoid recursive setter
  this._dest = dest;
  this._type = type;
  this._control_points = [];
  this.initControlPoints();
}

Edge.prototype = Object.create(Shape.prototype);

// define getter and setters
// setters eagerly maintains size, bbox and control points
// eager setters are efficient in this case because most
// edges in a polygon is not changing at every frame
Object.defineProperty(Edge.prototype, 'src', {
  get: function() {return this._src;},
  set: function(vertex) {
    this._src = vertex;
    this.initControlPoints();
  },
});

Object.defineProperty(Edge.prototype, 'dest', {
  get: function() {return this._dest;},
  set: function(vertex) {
    this._dest = vertex;
    this.initControlPoints();
  },
});

Object.defineProperty(Edge.prototype, 'type', {
  get: function() {return this._type;},
  set: function(newType) {
    this._type = newType;
    this.initControlPoints();
  },
});

Object.defineProperty(Edge.prototype, 'control_points', {
  get: function() {
    switch (this.type) {
      case EdgeTypes.LINE:
        this.initControlPoints();
    }
    return this._control_points;
  },
  set: function(points) {
    this._control_points = points;
  },
});

Object.defineProperty(Edge.prototype, 'length', {
  get: function() {return this.src.distanceTo(this.dest);},
});

Object.defineProperty(Edge.prototype, 'size', {
  get: function() {return 1 + this.control_points.length;},
});

Object.defineProperty(Edge.prototype, 'bbox', {
  get: function() {
    let points = [this.src, this.dest].concat(this.control_points);
    return {min: points.reduce(minVertex, points[0]),
      max: points.reduce(maxVertex, points[0])};
  },
});

// get the other vertex given v
Edge.prototype.getOther = function(v) {
  if (v.equals(this.src)) {
    return this.dest;
  }
  if (v.equals(this.dest)) {
    return this.src;
  }
  return -1;
};

// compute midpoints of vertices or control points on bezier curve
Edge.prototype.initControlPoints = function() {
  switch (this.type) {
    case EdgeTypes.LINE: {
      let midpoint = this.src.interpolate(this.dest, 1/2);
      midpoint.type = VertexTypes.MIDPOINT;
      this._control_points = [midpoint];
      break;
    }
    case EdgeTypes.BEZIER: {
      let control1 = this.src.interpolate(this.dest, 1/3);
      let control2 = this.src.interpolate(this.dest, 2/3);
      control1.type = VertexTypes.CONTROL_POINT;
      control2.type = VertexTypes.CONTROL_POINT;
      this._control_points = [control1, control2];
      break;
    }
  }
};

Edge.prototype.reverse = function() {
  let temp = this.src;
  this.src = this.dest;
  this.dest = temp;
  this.control_points = this.control_points.reverse();
};

// check whether the edge contains a point
Edge.prototype.contains = function(v) {
  // be careful if there is divide by 0
  let delta1 = (v.y - this.src.y) / (v.x - this.src.x);
  let delta2 = (this.dest.y - this.src.y) / (this.dest.x - this.src.x);
  let lambda = (v.y - this.src.y) / (this.dest.y - this.src.y);
  return delta1 === delta2 && 0 < lambda && lambda < 1;
};

// check whether the edge intersect with the target edge
Edge.prototype.intersectWith = function(e) {
  let det;
  let gamma;
  let lambda;
  det = (this.dest.x - this.src.x) * (e.dest.y - e.src.y)
    - (e.dest.x - e.src.x) * (this.dest.y - this.src.y);
  if (det === 0) { // parallel
    return this.contains(e.src) || this.contains(e.dest);
  } else {
    lambda = ((e.dest.y - e.src.y) * (e.dest.x - this.src.x)
      + (e.src.x - e.dest.x) * (e.dest.y - this.src.y)) / det;
    gamma = ((this.src.y - this.dest.y) * (e.dest.x - this.src.x)
      + (this.dest.x - this.src.x) * (e.dest.y - this.src.y)) / det;
    return (0 < lambda && lambda < 1) && (0 < gamma && gamma < 1);
  }
};

Edge.prototype.toJson = function() {
  let points = [];
  for (let controlPoint of this.control_points) {
    points.push(controlPoint.toJson());
  }
  return {
    src: this.src.toJson(),
    dest: this.dest.toJson(),
    type: this.type,
    control_points: points,
  };
};

Edge.fromJson = function(json) {
  let e = new Edge();
  e.src = Vertex.fromJson(json.src);
  e.dest = Vertex.fromJson(json.dest);
  e.type = json.type;
  let points = json.control_points;
  e.control_points = [];
  for (let controlPoint of points) {
    e.control_points.push(Vertex.fromJson(controlPoint));
  }
  return e;
};

// Reference safe Deep copy by serialization
Edge.prototype.copy = function() {
  return Edge.fromJson(this.toJson());
};

// Equality criteria for undirected edges
Edge.prototype.equals = function(e) {
  let vertexMatch = this.src.equals(e.src) && this.dest.equals(e.dest);
  let reverseMatch = this.src.equals(e.dest) && this.dest.equals(e.src);
  if (!(vertexMatch || reverseMatch)) {return false;}
  if (this.type != e.type) {return false;}
  if (this.size != e.size) {return false;}
  for (let i = 0; i < this.control_points.length; i++) {
    let point1 = this.control_points[i];
    let point2;
    if (reverseMatch) {
      point2 = e.control_points[this.control_points.length-i-1];
    } else {
      point2 = e.control_points[i];
    }
    if (!point1.equals(point2)) {
      return false;
    }
  }
  return true;
};

/**
 * Polygon Class, a data structure for Seg2d that holds
 * information about a polygon
 * @param {object []} vertices: object array holding
 * vertices of polygon
 */
function Polygon() {
  Shape.call(this);
  this.vertices = [];
  this.edges = [];
  this.closed = false;
}

Polygon.prototype = Object.create(Shape.prototype);

Object.defineProperty(Polygon.prototype, 'control_points', {
  get: function() {
    let points = [];
    for (let edge of this.edges) {
      points = points.concat(edge.control_points());
    }
    return points;
  },
});

Object.defineProperty(Polygon.prototype, 'bbox', {
  get: function() {
    let points = [];
    for (let edge of this.edges) {
      points = points.concat([edge.bbox.min, edge.bbox.max]);
    }
    return {min: points.reduce(minVertex, points[0]),
    max: points.reduce(maxVertex, points[0])};
  },
});

// wrap index
Polygon.prototype.idx = function(i) {
  return (i + this.vertices.length) % this.vertices.length;
};

// return centroid of the polygon
Polygon.prototype.centroid = function() {
  return this.vertices.reduce(meanVertex, new Vertex());
};

// check whether a polygon is valid
Polygon.prototype.isValid = function() {
  return !(this.isSmall() || this.isSelfIntersect());
};

// check whether a polygon is too small, return true if
// any edge in the polygon is <= 20
Polygon.prototype.isSmall = function() {
  let small = true;
  for (let edge of this.edges) {
    if (edge.length > 20) {
      small = false;
    }
  }
  return small;
};

// return true is any two edges inside a polygon intersect with each other
Polygon.prototype.isSelfIntersect = function() {
  let intersect = false;
  for (let i = 0; i < this.vertices.length; i++) {
    for (let j = i+1; j < this.vertices.length; j++) {
      if (this.edges[i].intersectWith(this.edges[j])) {
        intersect = true;
      }
    }
  }
  return intersect;
};

Polygon.prototype.alignEdges = function() {
  for (let i = 0; i < this.edges.length; i++) {
    if (this.vertices[i].equals(this.edges[i].dest)) {
      this.edges[i].reverse();
    }
  }
};

Polygon.prototype.reverse = function() {
  this.vertices = this.vertices.reverse();
  this.vertices.unshift(this.vertices.pop());
  this.edges = this.edges.reverse();
  for (let i = 0; i < this.edges.length; i++) {
    this.edges[i].reverse();
  }
};

Polygon.prototype.closePath = function() {
  this.closed = true;
};

Polygon.prototype.isClosed = function() {
  return this.closed;
};

/**
 * Insert vertex to polygon at given position i
 * Assuming prev and next vertices are connected by line, not bezier
 * @param {int} i: id for the polygon
 * @param {object} pt: the new vertex to be inserted
 */
Polygon.prototype.insertVertex = function(i, pt) {
  this.vertices.splice(i, 0, pt);
  if (this.vertices.length > 1) {
    let edge1 = new Edge(
      this.vertices[this.idx(i-1)], this.vertices[i]
    );
    let edge2 = new Edge(
      this.vertices[i], this.vertices[this.idx(i+1)]
    );
    this.edges.splice(this.idx(i-1), 1, edge1);
    this.edges.splice(i, 0, edge2);
  } else {
    this.edges = [];
  }
};

/**
 * Append new vertex to end of vertex sequence of the polygon
 * @param {object} pt: the new vertex to be inserted
 */
Polygon.prototype.pushVertex = function(pt) {
  this.insertVertex(this.vertices.length, pt);
};

/**
 * Delete vertex from polygon at given position i
 * @param {int} i: id for the polygon
 */
Polygon.prototype.deleteVertex = function(i) {
  this.vertices.splice(i, 1);
  if (this.vertices.length > 1) {
    let edge = new Edge(
      this.vertices[this.idx(i-1)], this.vertices[this.idx(i)]
    );
    this.edges.splice(i, 1);
    this.edges.splice(this.idx(i-1), 1, edge);
  } else {
    this.edges = [];
  }
};

Polygon.prototype.popVertex = function() {
  this.deleteVertex(this.vertices.length-1);
};

// find index of a vertex
Polygon.prototype.indexOf = function(v) {
  let index = -1;
  for (let i = 0; i < this.vertices.length; i++) {
    if (v.equals(this.vertices[i])) {
      index = i;
      break;
    }
  }
  return index;
};

// TODO: insert path at arbitrary position, default to shorter path
Polygon.prototype.pushPath = function(i, targetPoly,
                                      vStart, vEnd, toggle=false) {
  return toggle; // dummy
};

// push a new path to the end of this polygon
Polygon.prototype.pushPath = function(targetPoly, vStart, vEnd, toggle=false) {
  if (toggle) {targetPoly.reverse();}
  let startIndex = targetPoly.indexOf(vStart);
  let endIndex = targetPoly.indexOf(vEnd);
  if (startIndex === -1 || endIndex === -1) {return;}
  if (endIndex < startIndex) {
    endIndex = endIndex + targetPoly.vertices.length;
  }
  this.edges.pop();
  this.edges.push(new Edge(this.vertices[this.vertices.length-1], vStart));
  for (let i = startIndex; i < endIndex; i++) {
    this.edges.push(targetPoly.edges[targetPoly.idx(i)]);
  }
  this.edges.push(new Edge(vEnd, this.vertices[0]));
  for (let i = startIndex; i <= endIndex; i++) {
    this.vertices.push(targetPoly.vertices[targetPoly.idx(i)]);
  }
};

Polygon.prototype.toJson = function() {
  let vertexJsons = [];
  let edgeJsons = [];
  for (let i = 0; i < this.edges.length; i++) {
    vertexJsons[i] = this.vertices[i].toJson();
    edgeJsons[i] = this.edges[i].toJson();
  }
  return {
    vertices: vertexJsons,
    edges: edgeJsons,
  };
};

Polygon.fromJson = function(json) {
  let poly = new Polygon();
  let vertexJsons = json.vertices;
  let edgeJsons = json.edges;
  for (let vertexJson of vertexJsons) {
    poly.vertices.push(Vertex.fromJson(vertexJson));
  }
  for (let edgeJson of edgeJsons) {
    poly.edges.push(Edge.fromJson(edgeJson));
  }
  return poly;
};

// Reference safe deep copy by serialization
Polygon.prototype.copy = function() {
  return Polygon.fromJson(this.toJson());
};

Polygon.prototype.equals = function(p) {
  let numVertexMatch = this.vertices.length === p.vertices.length;
  if (!numVertexMatch) {return false;}
  let numEdgeMatch = this.edges.length === p.edges.length;
  if (!numEdgeMatch) {return false;}
  let vertex = this.vertices[0];
  let index = p.indexOf(vertex);
  if (index === -1) {return false;}
  let reversed = this.vertices[1].equals(p.vertices[p.idx(index-1)]);
  let direction = reversed ? -1 : 1;
  let offset = reversed ? -1 : 0;
  for (let i = 0; i < this.vertices.length; i++) {
    if (!this.vertices[i].equals(p.vertices[p.idx(index + direction*i)])) {
      return false;
    }
  }
  for (let i = 0; i < this.edges.length; i++) {
    if (!this.edges[i].equals(p.edges[p.idx(index + direction*i + offset)])) {
      return false;
    }
  }
  return true;
};


