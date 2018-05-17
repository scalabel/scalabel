/**
 * Edge Class
 * @param {int []} src: source vertex
 * @param {int []} dest: destination vertex
 * @param {string} type: connection type, line or bezier
 */
function Edge(src, dest, type='line') {
  this.src = src;
  this.dest = dest;
  this.type = type;
  this.size = 0;
  this.control_points = [];
  this.recomputeControlPoints()
  this.bbox = [0,0,0,0];
  this.recomputeBbox();
}

Edge.prototype.recomputeControlPoints = function() {
  if (this.type === 'line') {
    this.control_points = [[
      Math.round((this.src[0] + this.dest[0]) / 2),
      Math.round((this.src[1] + this.dest[1]) / 2)
    ]];
  }
  if (this.type === 'bezier') {
    let control1 = [
      Math.round((this.src[0] * 2 + this.dest[0]) / 3),
      Math.round((this.src[1] * 2 + this.dest[1]) / 3)
    ];
    let control2 = [
      Math.round((this.src[0] + this.dest[0] * 2) / 3),
      Math.round((this.src[1] + this.dest[1] * 2) / 3)
    ];
    this.control_points = [control1, control2];
  }
  this.size = 1 + this.control_points.length;
}

Edge.prototype.recomputeBbox = function() {
  this.bbox = [
    Math.min(this.src[0], this.dest[0]),
    Math.min(this.src[1], this.dest[1]),
    Math.max(this.src[0], this.dest[0]),
    Math.max(this.src[1], this.dest[1])
  ];
  for (let i = 0; i < this.control_points.length; i++) {
    this.bbox = [
      Math.min(this.bbox[0], this.control_points[i][0]),
      Math.min(this.bbox[1], this.control_points[i][1]),
      Math.max(this.bbox[2], this.control_points[i][0]),
      Math.max(this.bbox[3], this.control_points[i][1])
    ];
  }
}

Edge.prototype.setSrc = function(pt) {
  this.src = pt;
  if (this.type === 'line') {
    this.recomputeControlPoints()
  }
  this.recomputeBbox();
}

Edge.prototype.setDest = function(pt) {
  this.dest = pt;
  if (this.type === 'line') {
    this.recomputeControlPoints()
  }
  this.recomputeBbox();
}

Edge.prototype.setType = function(type) {
  this.type = type;
  this.recomputeControlPoints();
  this.recomputeBbox();
}

Edge.prototype.reverse = function() {
  let temp = this.src;
  this.src = this.dest;
  this.dest = temp;
  this.control_points = this.control_points.reverse();
}

/**
 * draw an edge, assuming mouse already at src,
 * need to call beginPath and moveTo before
 * drawing the first edge in poly, and closePath
 * after last poly
 */
Edge.prototype.draw = function(context) {
  if (this.type === 'line') {
    context.lineTo(this.dest[0], this.dest[1]);
  }
  if (this.type === 'bezier') {
    context.bezierCurveTo(
      this.control_points[0][0], this.control_points[0][1],
      this.control_points[1][0], this.control_points[1][1],
      this.dest[0], this.dest[1]
    );
  }
}


Edge.prototype.toJson = function() {
  return {
    'src': JSON.stringify(this.src),
    'dest': JSON.stringify(this.dest),
    'type': this.type,
    'control_points': JSON.stringify(this.control_points),
    'bbox': JSON.stringify(this.bbox),
    'size': this.size
  }
}

Edge.fromJson = function(data) {
  let e = new Edge([0,0],[0,0]);
  e.src = JSON.parse(data['src']);
  e.dest = JSON.parse(data['dest']);
  e.type = data['type'];
  e.control_points = JSON.parse(data['control_points']);
  e.bbox = JSON.parse(data['bbox']);
  e.size = data['size'];
  return e;
}

/**
 * Polygon Class
 * @param {int} id: id for the polygon
 * @param {int [][]} p: 2D array holding vertex coordinates of polygon
 */
function Polygon(id, vertices=[]) {
  this.id = id;
  this.vertices = vertices;
  this.edges = Array(vertices.length);
  for (let i = 0; i < this.vertices.length; i++) {
    this.edges[i] = new Edge(
      this.vertices[i],
      this.vertices[this.idx(i+1)]
    );
  }
  this.bbox = [0,0,0,0];
  this.recomputeBbox();
}

// wrap index
Polygon.prototype.idx = function(i) {
  return (i + this.vertices.length) % this.vertices.length;
}

Polygon.prototype.reverse = function() {
  this.vertices = this.vertices.reverse();
  this.vertices.unshift(this.vertices.pop())
  this.edges = this.edges.reverse();
  for (let i = 0; i < this.edges.length; i++) {
    this.edges[i].reverse();
  }
}

/**
 * Set edge type at position i and updates control points
 * @param {int} i: position to update => edge between vertex i and (i+1)
 * @param {string} type: edge type (line vs. bezier)
 */
Polygon.prototype.setEdgeType = function(i, type) {
  this.edges[i].setType(type);
}

/**
 * Insert vertex to polygon at given position i
 * Assuming prev and next vertices are connected by line, not bezier
 * @param {int} id: id for the polygon
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
  this.recomputeBbox();
}

/**
 * Append new vertex to end of vertex sequence of the polygon
 * @param {int} id: id for the polygon
 */
Polygon.prototype.pushVertex = function(pt) {
  this.insertVertex(this.vertices.length, pt);
}

/**
 * Delete vertex from polygon at given position i
 * @param {int} id: id for the polygon
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
  this.recomputeBbox();
}

Polygon.prototype.popVertex = function() {
  this.deleteVertex(this.vertices.length-1);
}

/**
 * modify vertex of polygon at given position i
 * @param {int} id: id for the polygon
 */
Polygon.prototype.moveHandle = function(new_pt, handleNo) {
  let [edge_id, ctrl_point_id] = this._getHandle(handleNo);

  if (ctrl_point_id === 0) { // vertex is selected
    this.vertices[edge_id] = new_pt;
    this.edges[edge_id].setSrc(new_pt);
    this.edges[
    (edge_id+this.vertices.length-1) % this.vertices.length
      ].setDest(new_pt);
  } else if (this.edges[edge_id].type === 'bezier') {
    this.edges[edge_id].control_points[ctrl_point_id-1] = new_pt;
  } else if (this.edges[edge_id].type === 'line') {
    this.insertVertex(edge_id+1, new_pt);
  }
  this.recomputeBbox();
}

Polygon.prototype.recomputeBbox = function() {
  this.bbox = [0,0,0,0];
  if (this.vertices.length > 0) {
    this.bbox = [
      this.vertices[0][0], this.vertices[0][1],
      this.vertices[0][0], this.vertices[0][1]
    ];
  }
  for (let i = 0; i < this.edges.length; i++) {
    this.bbox[0] = Math.min(this.edges[i].bbox[0], this.bbox[0]);
    this.bbox[1] = Math.min(this.edges[i].bbox[1], this.bbox[1]);
    this.bbox[2] = Math.max(this.edges[i].bbox[2], this.bbox[2]);
    this.bbox[3] = Math.max(this.edges[i].bbox[3], this.bbox[3]);
  }
};

/**
 * Get hidden color for a vertex/control point
 * @param {int} vertex_id: edge index
 * @param {int} point_selector: 0 for vertex,
 * always 1 for line control point, 1/2 for bezier control point
 */
Polygon.prototype.hiddenStyleColor = function(edge_id, ctrl_point_id) {
  let handleNo = 1 + ctrl_point_id;
  for (let i = 0; i < edge_id; i++) {
    handleNo  = handleNo + this.edges[i].size;
  }
  return 'rgb(' + [this.id & 255, (this.id >> 8) & 255,
    handleNo].join(',') + ')';
}

/**
 * Recover selected vertex_id and point_selector from hiddenStyleColor
 * @param {int} handleNo: handle number
 */
Polygon.prototype._getHandle = function(handleNo) {
  if (handleNo === 0) {
    return [-1, -1];
  }

  let ctrl_point_id = handleNo - 1;
  let edge_id = 0;
  while (ctrl_point_id > 0) {
    if (ctrl_point_id < this.edges[edge_id].size) {break}
    ctrl_point_id = ctrl_point_id - this.edges[edge_id].size;
    edge_id = edge_id + 1;
  }

  return [edge_id, ctrl_point_id];
}

Polygon.prototype.drawPolygon = function(context) {
  context.beginPath();
  context.moveTo(this.vertices[0][0], this.vertices[0][1]);
  for (let j = 1; j < this.edges.length; j++) {
    this.edges[j].draw(context);
  }
  context.closePath();

  context.fillStyle = '#f00';
  ctx.fill();
}

Polygon.prototype.drawHiddenPolygon = function() {
  context.beginPath();
  context.moveTo(this.vertices[0][0], this.vertices[0][1]);
  for (let j = 1; j < this.edges.length; j++) {
    this.edges[j].draw(context);
  }
  context.closePath();

  context.fillStyle = this.hiddenStyleColor(-1,-1);
  ctx.fill();
}

Polygon.prototype.drawHandle = function(handleNo) {

}

Polygon.prototype.drawHandles = function() {

}

Polygon.prototype.drawHiddenHandle = function(hiddenCtx, handleNo, labelIndex) {

}

Polygon.prototype.drawHiddenHandles = function() {

}

Polygon.prototype.drawTag = function() {

}

Polygon.prototype.redraw = function(mainCtx, hiddenCtx, selectedBox, resizing,
                                    hoverBox, hoverHandle, labelIndex) {

}

Polygon.prototype.toJson = function() {
  let edge_jsons = [];
  for (let i = 0; i < this.edges.length; i++) {
    edge_jsons[i] = this.edges[i].toJson();
  }
  return {
    'id':this.id,
    'vertices':this.vertices,
    'edges':edge_jsons,
    'bbox':this.bbox
  };
}

Polygon.fromJson = function(data) {
  let poly = new Polygon(0);
  poly.id = data['id'];
  poly.vertices = data['vertices'];
  poly.edges = Array(data['edges'].length);
  for (let i = 0; i < data['edges'].length; i++) {
    poly.edges[i] = Edge.fromJson(data['edges'][i]);
  }
  poly.bbox = data['bbox'];
  return poly;
}

module.exports = Polygon, Edge;