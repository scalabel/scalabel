const [Vertex, Edge, Polygon, VertexTypes, EdgeTypes] = require('../polygon');

/**
 * Random Integer Generator
 * @param {int} min: min value
 * @param {int} max: max value
 * @return {int}: random integer
 */
function randomInteger(min=1, max=1000) {
  return Math.floor(Math.random() * (max - min) ) + min;
}

/**
 * Random Vertex Generator
 * @return {object}: random vertex
 */
function randomVertex() {
  return new Vertex(randomInteger(), randomInteger());
}

/**
 * Convert list of coordinates to list of vertices
 * @param {object []} list: list of 2-tuples containing vertex coordinates
 * @return {object []}: list of vertices
 */
function list2Vertices(list) {
  let vertices = [];
  for (let tuple of list) {
    let [x, y] = tuple;
    vertices.push(new Vertex(x, y));
  }
  return vertices;
}

describe('Vertex Object Tests', function() {
  it('Vertex constructor and getter', function() {
    let x = randomInteger();
    let y = randomInteger();
    let vertex = new Vertex(x, y);
    expect(vertex.x).toBe(x);
    expect(vertex.y).toBe(y);
    expect(vertex.type).toBe(VertexTypes.VERTEX);
  });
  it('Vertex setter', function() {
    let x = randomInteger();
    let y = randomInteger();
    let vertex = new Vertex();
    vertex.xy = [x, y];
    expect(vertex.x).toBe(x);
    expect(vertex.y).toBe(y);
  });
  it('Vertex reference', function() {
    let vertex = randomVertex();
    let vertex2 = vertex;
    let vertex3 = vertex.copy();
    vertex2.xy = [0, 0];
    expect(vertex).toBe(vertex2);
    expect(vertex).not.toBe(vertex3);
  });
  it('Vertex interpolation', function() {
    let vertex = randomVertex();
    let vertex2 = randomVertex();
    let midPoint = vertex.interpolate(vertex2, 1/2);
    let x = (vertex.x + vertex2.x) / 2;
    let y = (vertex.y + vertex2.y) / 2;
    expect(midPoint.x).toBe(x);
    expect(midPoint.y).toBe(y);
  });
  it('Vertex distance', function() {
    let vertex = randomVertex();
    let vertex2 = randomVertex();
    let distance = vertex.distanceTo(vertex2);
    let a = vertex.x - vertex2.x;
    let b = vertex.y - vertex2.y;
    expect(distance * distance).toBeCloseTo(a * a + b * b);
  });
  it('Vertex copy', function() {
    let vertex = randomVertex();
    let newVertex = vertex.copy();
    expect(vertex.equals(newVertex)).toBe(true);
  });
  it('Vertex serialization', function() {
    let vertex = randomVertex();
    let newVertex = vertex.copy();
    expect(vertex.equals(newVertex)).toBe(true);
  });
});

/**
 * Random Edge Generator
 * @param {bool} randomizeType: whether to randomize edge type
 * @return {object}: random edge
 */
function randomEdge(randomizeType=false) {
  let e = new Edge(randomVertex(), randomVertex());
  if (Math.random() > 0.5 && randomizeType) {
    e.type = EdgeTypes.BEZIER;
  }
  return e;
}

describe('Edge Object Tests', function() {
  it('Edge constructor', function() {
    let v1 = new Vertex(2, 5);
    let v2 = new Vertex(5, 2);
    let e = new Edge(v1, v2);
    // getters
    expect(e.src).toBe(v1);
    expect(e.dest).toBe(v2);
    expect(e.getOther(e.dest)).toBe(v1);
    expect(e.getOther(e.src)).toBe(v2);
    // control point, line
    expect(e.control_points[0].equals(new Vertex(3.5, 3.5))).toBe(true);
    // bbox, line
    expect(e.bbox.min.equals(new Vertex(2, 2))).toBe(true);
    expect(e.bbox.max.equals(new Vertex(5, 5))).toBe(true);
    // set type
    e.type = EdgeTypes.BEZIER;
    expect(e.size).toBe(3);
    // control point, bezier
    expect(e.control_points[0].equals(new Vertex(3, 4))).toBe(true);
    expect(e.control_points[1].equals(new Vertex(4, 3))).toBe(true);
    // bbox, bezier
    e.control_points[0].xy = [6, 6];
    expect(e.bbox.min.equals(new Vertex(2, 2))).toBe(true);
    expect(e.bbox.max.equals(new Vertex(6, 6))).toBe(true);
    // length
    expect(e.length).toBeCloseTo(4.24264069);
    // setters
    e.src = v2;
    e.dest = v1;
    expect(e.control_points[0].equals(new Vertex(4, 3))).toBe(true);
    expect(e.control_points[1].equals(new Vertex(3, 4))).toBe(true);
  });
  it('Edge contain', function() {
    let vv1 = new Vertex(0, 0);
    let vv2 = new Vertex(2, 2);
    let vv3 = new Vertex(1, 1);
    let vv4 = new Vertex(2, -10);
    let e = new Edge(vv1, vv2);
    expect(e.contains(vv3)).toBe(true);
    expect(e.contains(vv4)).toBe(false);
  });
  it('Edge intersect', function() {
    let vv1 = new Vertex(0, 0);
    let vv2 = new Vertex(2, 2);
    let vv3 = new Vertex(0, 10);
    let vv4 = new Vertex(2, -10);
    let e1 = new Edge(vv1, vv2);
    let e2 = new Edge(vv3, vv4);
    expect(e1.intersectWith(e2)).toBe(true);
  });
  it('Edge share vertex', function() {
    let vv1 = new Vertex(0, 0);
    let vv2 = new Vertex(2, 2);
    let vv3 = new Vertex(1, 1);
    let vv4 = new Vertex(2, -2);
    let e1 = new Edge(vv1, vv2);
    let e2 = new Edge(vv3, vv4);
    expect(e1.intersectWith(e2)).toBe(false);
  });
  it('Edge overlay', function() {
    let vv1 = new Vertex(0, 0);
    let vv2 = new Vertex(2, 2);
    let vv3 = new Vertex(1, 1);
    let vv4 = new Vertex(3, 3);
    let e1 = new Edge(vv1, vv2);
    let e2 = new Edge(vv3, vv4);
    expect(e1.intersectWith(e2)).toBe(true);
  });
  it('Edge equals', function() {
    let v1 = randomVertex();
    let v2 = randomVertex();
    let e1 = new Edge(v1, v2);
    let e2 = new Edge(v1, v2);
    expect(e1.equals(e2)).toBe(true);
    e2.reverse();
    expect(e1.equals(e2)).toBe(true);
    e1.type = EdgeTypes.BEZIER;
    expect(e1.equals(e2)).toBe(false);
    e2.type = EdgeTypes.BEZIER;
    expect(e1.equals(e2)).toBe(true);
    e1.control_points[0].xy = [10, 10];
    expect(e1.equals(e2)).toBe(false);
    e2.control_points[1].xy = [10, 10];
    expect(e1.equals(e2)).toBe(true);
  });
  it('Edge serialize', function() {
    // line
    let e1 = randomEdge();
    let e2 = e1.copy();
    expect(e1.equals(e2)).toBe(true);
    // bezier
    let e3 = randomEdge();
    e3.type = EdgeTypes.BEZIER;
    e3.control_points[0].xy = [10, 10];
    let e4 = e3.copy();
    expect(e3.equals(e4)).toBe(true);
  });
});

/**
 * Random Polygon Generator
 * @param {int} size: number of vertices
 * @return {object}: random polygon
 */
function randomPolygon(size=255) {
  let p = new Polygon();
  for (let i = 0; i < size; i++) {
    p.pushVertex(randomVertex());
  }
  return p;
}

/**
 * Convert list of vertices to a polygon
 * @param {object []} vertices: list of vertices
 * @return {object}: polygon
 */
function vertices2Polygon(vertices) {
  let p = new Polygon();
  for (let vertex of vertices) {
    p.pushVertex(vertex);
  }
  return p;
}

/**
 * Convert list of coordinates to a polygon
 * @param {object []} list: list of vertices
 * @return {object}: polygon
 */
function list2Polygon(list) {
  let vertices = list2Vertices(list);
  return vertices2Polygon(vertices);
}

describe('Polygon Object Tests', function() {
  it('Polygon Constructor', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    expect(p.bbox.min.equals(new Vertex(0, 0))).toBe(true);
    expect(p.bbox.max.equals(new Vertex(8, 8))).toBe(true);
    expect(p.centroid().equals(new Vertex(4, 4))).toBe(true);
  });
  it('Polygon align edge', function() {
    let p = randomPolygon();
    for (let edge of p.edges) {
      if (Math.random() > 0.5) {
        edge.reverse();
      }
    }
    let edgeAligned = true;
    for (let i = 0; i < p.edges.length; i++) {
      edgeAligned = edgeAligned &&
        p.edges[i].src.equals(p.vertices[i]) &&
        p.edges[i].dest.equals(p.vertices[p.idx(i+1)]);
      if (!edgeAligned) {break;}
    }
    expect(edgeAligned).toBe(false);
    p.alignEdges();
    edgeAligned = true;
    for (let i = 0; i < p.edges.length; i++) {
      edgeAligned = edgeAligned &&
        p.edges[i].src.equals(p.vertices[i]) &&
        p.edges[i].dest.equals(p.vertices[p.idx(i+1)]);
    }
    expect(edgeAligned).toBe(true);
  });
  it('Polygon reverse', function() {
    let p = randomPolygon();
    p.reverse();
    let p2 = p.copy();
    p.reverse();
    expect(p.equals(p2)).toBe(true);
  });
  it('Polygon Serialization', function() {
    let p = randomPolygon();
    let p2 = p.copy();
    expect(p.equals(p2)).toBe(true);
  });
  it('Polygon pushPath', function() {
    let p1 = list2Polygon([
      [0, 0], [0, 4], [2, 4], [2, 3], [2, 2], [2, 1], [2, 0]]);
    let p2 = list2Polygon([
      [4, 1], [6, 1], [6, 4], [4, 4]]);
    let p3 = p2.copy();
    let pTarget = list2Polygon([
      [4, 1], [6, 1], [6, 4], [4, 4], [2, 4], [2, 3], [2, 2], [2, 1]]);
    let pTarget2 = list2Polygon([
      [4, 1], [6, 1], [6, 4], [4, 4], [2, 4],
      [0, 4], [0, 0], [2, 0], [2, 1]]);
    let vStart = new Vertex(2, 4);
    let vEnd = new Vertex(2, 1);
    p2.pushPath(p1, vStart, vEnd);
    expect(p2.equals(pTarget)).toBe(true);
    p3.pushPath(p1, vStart, vEnd, true);
    expect(p3.equals(pTarget2)).toBe(true);
  });
});

describe('Shared reference behavior Tests', function() {
  it('Shared vertex movement', function() {
    let [v1, v2, v3, v4, v5, v6] = list2Vertices([[0, 0], [2, 0],
      [4, 0], [4, 4], [2, 4], [0, 4]]);
    let v7 = new Vertex(2, 2);
    let p1 = vertices2Polygon([v1, v2, v5, v6]);
    let p2 = vertices2Polygon([v2, v3, v4, v5]);
    let p3 = vertices2Polygon([v1, v2, v7, v6]);
    let p4 = vertices2Polygon([v2, v3, v4, v7]);
    p1.reverse();
    v5.xy = [2, 2]; // move v5
    expect(p1.equals(p3)).toBe(true);
    expect(p2.equals(p4)).toBe(true);
  });
});
