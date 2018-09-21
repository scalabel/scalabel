import {Shape, Vertex, Edge, Path, Polygon,
  VertexTypes, EdgeTypes} from '../shape';
import {idx} from '../utils';

/**
 * Random Integer Generator
 * @param {int} min: min value
 * @param {int} max: max value
 * @return {int}: random integer
 */
function randomInteger(min = 1, max = 1000) {
  return Math.floor(Math.random() * (max - min)) + min;
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

describe('Shape Object Tests', function() {
  it('Shape', function() {
    let s = new Shape(); // dummy
    expect(s === s).toBe(true);
  });
});

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
    let midPoint = vertex.interpolate(vertex2, 1 / 2);
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
function randomEdge(randomizeType = false) {
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
    // control point, line
    expect(e.control_points[0].equals(new Vertex(3.5, 3.5))).toBe(true);
    expect(e.size).toBe(2);
    // bbox, line
    expect(e.bbox.min.x).toBe(2);
    expect(e.bbox.min.y).toBe(2);
    expect(e.bbox.max.x).toBe(5);
    expect(e.bbox.max.y).toBe(5);
    // set type
    e.type = EdgeTypes.BEZIER;
    expect(e.size).toBe(3);
    // control point, bezier
    expect(e.control_points[0].equals(new Vertex(3, 4))).toBe(true);
    expect(e.control_points[1].equals(new Vertex(4, 3))).toBe(true);
    // bbox, bezier
    e.control_points[0].xy = [6, 6];
    expect(e.bbox.min.x).toBe(2);
    expect(e.bbox.min.y).toBe(2);
    expect(e.bbox.max.x).toBe(6);
    expect(e.bbox.max.y).toBe(6);
    // length
    expect(e.length).toBeCloseTo(4.24264069);
    // setters
    e.reverse();
    e.initControlPoints(); // this is important!!!
    expect(e.control_points[0].equals(new Vertex(4, 3))).toBe(true);
    expect(e.control_points[1].equals(new Vertex(3, 4))).toBe(true);
  });
  it('Edge drag', function() {
    let vv1 = new Vertex(0, 0);
    let vv2 = new Vertex(2, 2);
    let e = new Edge(vv1, vv2);
    e.control_points[0].index = 100;
    let expectedMidPoint = new Vertex(0, 0);
    e.src.xy = [-2, -2];
    expect(e.control_points[0].equals(expectedMidPoint)).toBe(true);
    expect(e.control_points[0].index).toBe(100);
  });
  it('Edge contain', function() {
    let vv1 = new Vertex(0, 0);
    let vv2 = new Vertex(2, 2);
    let vv3 = new Vertex(1, 1);
    let vv4 = new Vertex(2, -10);
    let e = new Edge(vv1, vv2);
    // does not contain source vertex
    expect(e.contains(vv1)).toBe(false);
    // does not contain destination vertex
    expect(e.contains(vv2)).toBe(false);
    // does contain the midpoint
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
    let e3 = new Edge(vv2, vv3);
    expect(e1.intersectWith(e2)).toBe(true);
    expect(e1.intersectWith(e3)).toBe(false);
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
    let e3 = new Edge(vv2, vv3);
    let e4 = new Edge(vv1, vv4);
    // overlaid edges intersect
    expect(e1.intersectWith(e2)).toBe(true);
    // an edge intersects with itself
    expect(e1.intersectWith(e1)).toBe(true);
    // and edge intersects with part of itself
    expect(e1.intersectWith(e3)).toBe(true);
    // an edge intersects with an parallel edge that is longer
    expect(e3.intersectWith(e4)).toBe(true);
  });
  it('Edge equals', function() {
    let v1 = randomVertex();
    let v2 = randomVertex();
    let e1 = new Edge(v1, v2);
    let e2 = new Edge(v1, v2);
    let e3 = new Edge(v2, v1);
    let e4 = e3.copy();
    expect(e1.equals(e2)).toBe(true);
    expect(e1.equals(e3)).toBe(true);
    expect(e1.equals(e4)).toBe(true);
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
 * Random Path Generator
 * @param {int} size: number of vertices
 * @return {object}: random path
 */
function randomPath(size = 255) {
  let p = new Path();
  for (let i = 0; i < size; i++) {
    p.pushVertex(randomVertex());
  }
  return p;
}

/**
 * Convert list of vertices to a path
 * @param {object []} vertices: list of vertices
 * @return {object}: path
 */
function vertices2Path(vertices) {
  let p = new Path();
  for (let vertex of vertices) {
    p.pushVertex(vertex);
  }
  return p;
}

/**
 * Convert list of coordinates to a path
 * @param {object []} list: list of vertices
 * @return {object}: path
 */
function list2Path(list) {
  let vertices = list2Vertices(list);
  return vertices2Path(vertices);
}

describe('Path Object Tests', function() {
  it('Path constructor', function() {
    let p = list2Path([[0, 0], [8, 0], [8, 8], [0, 8]]);
    expect(p.bbox.min.x).toBe(0);
    expect(p.bbox.min.y).toBe(0);
    expect(p.bbox.max.x).toBe(8);
    expect(p.bbox.max.y).toBe(8);
    // expect(p.centroid().equals(new Vertex(4, 4))).toBe(true);
  });
  it('Path copy reverse equal', function() {
    let p = randomPath();
    let p2 = p.copy();
    expect(p.equals(p2)).toBe(true);
    expect(p2 instanceof Path).toBe(true);
    p2.reverse();
    expect(p.equals(p2)).toBe(true);
  });
  it('Path insert vertex', function() {
    let p = list2Path([[0, 0], [8, 0]]);
    let pTarget = list2Path([[0, 0], [8, 0], [8, 8], [0, 8]]);
    p.pushVertex(new Vertex(0, 8));
    p.insertVertex(2, new Vertex(8, 8));
    expect(p.equals(pTarget)).toBe(true);
  });
  it('Path delete vertex', function() {
    let p = list2Path([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let pTarget = list2Path([[0, 0], [8, 0]]);
    let pTarget2 = list2Path([[8, 0]]);
    let pTarget3 = new Path();
    p.deleteVertex(2);
    p.popVertex();
    expect(p.equals(pTarget)).toBe(true);
    p.deleteVertex(0);
    expect(p.equals(pTarget2)).toBe(true);
    p.popVertex();
    expect(p.equals(pTarget3)).toBe(true);
  });
});

/**
 * Random Polygon Generator
 * @param {int} size: number of vertices
 * @return {object}: random polygon
 */
function randomPolygon(size = 255) {
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
  p.endPath();
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
  it('Polygon constructor', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    expect(p.bbox.min.x).toBe(0);
    expect(p.bbox.min.y).toBe(0);
    expect(p.bbox.max.x).toBe(8);
    expect(p.bbox.max.y).toBe(8);
    // expect(p.centroid().equals(new Vertex(4, 4))).toBe(true);
  });
  it('Polygon self intersect floating point', function() {
    let p = list2Polygon([
      [675.2340425531914, 467.22099921043883],
      [937.1914893617022, 467.22099921043883],
      [845.2765957446809, 618.8805736785239]]);
    expect(p.isSelfIntersect()).toBe(false);
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
          p.edges[i].dest.equals(p.vertices[idx(i + 1, p.vertices.length)]);
      if (!edgeAligned) {
        break;
      }
    }
    expect(edgeAligned).toBe(false);
    p.alignEdges();
    edgeAligned = true;
    for (let i = 0; i < p.edges.length; i++) {
      edgeAligned = edgeAligned &&
          p.edges[i].src.equals(p.vertices[i]) &&
          p.edges[i].dest.equals(p.vertices[idx(i + 1, p.vertices.length)]);
    }
    expect(edgeAligned).toBe(true);
  });
  it('Polygon reverse', function() {
    let p = randomPolygon();
    p.reverse();
    let p2 = p.copy(); // p2 is the reverse of the original p
    expect(p.equals(p2)).toBe(true);
    p.reverse(); // p reverses back to its original structure
    expect(p.equals(p2)).toBe(true);

    let p3 = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let p4 = list2Polygon([[0, 0], [0, 8], [8, 8], [8, 0]]);
    for (let i = 0; i < p3.edges.length; i++) {
      expect((p3.edges[i]).equals(p4.edges[i])).toBe(false);
    }
    p3.reverse();
    for (let i = 0; i < p3.edges.length; i++) {
      expect((p3.vertices[i]).equals(p4.vertices[i])).toBe(true);
      expect((p3.edges[i]).equals(p4.edges[i])).toBe(true);
    }
  });
  it('Polygon insert vertex', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let p2 = list2Polygon([[0, 0], [8, 0], [2, 2], [8, 8], [0, 8]]);
    let v = new Vertex(2, 2);
    p.insertVertex(2, v);
    p.insertVertex(-1, v); // do nothing if i is out of index
    p.insertVertex(10, v);
    expect((p.vertices[2]).equals(v)).toBe(true);
    expect(p.equals(p2)).toBe(true);
    // insert vertex to the start of the list
    p.insertVertex(0, v);
    expect((p.vertices[0]).equals(v)).toBe(true);
    // insert vertex to the last position
    p.insertVertex(p.vertices.length, v);
    expect((p.vertices[p.vertices.length - 1]).equals(v)).toBe(true);
  });
  it('Polygon midpoint to vertex', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let targetP = list2Polygon([[0, 0], [4, 0], [8, 0], [8, 8], [0, 8]]);

    p.midpointToVertexWithEdgeIndex(0);
    expect(p.equals(targetP)).toBe(true);
  });
  it('Polygon push vertex', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let p2 = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8], [2, 2]]);
    let v = new Vertex(2, 2);

    p.pushVertex(v);
    expect((p.vertices[4]).equals(v)).toBe(true);
    expect(p.equals(p2)).toBe(true);
  });
  it('Polygon delete vertex', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let p2 = list2Polygon([[0, 0], [8, 0], [0, 8]]);
    p.deleteVertex(2);
    p.deleteVertex(-1); // do nothing if i is out of index
    p.deleteVertex(10);
    expect(p.equals(p2)).toBe(true);
  });
  it('Polygon pop vertex', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let p2 = list2Polygon([[0, 0], [8, 0], [8, 8]]);
    p.popVertex();
    expect(p.equals(p2)).toBe(true);
  });
  it('Polygon indexOf', function() {
    let p = list2Polygon([[0, 0], [8, 0], [8, 8], [0, 8]]);
    let v = new Vertex(0, 0);
    expect(p.indexOf(v)).toBe(0);
    expect(p.indexOf(new Vertex(100, 0))).toBe(-1);
  });
  it('Polygon getPathBetween', function() {
    let [v1, v2, v3, v4] = list2Vertices([
      [0, 0], [0, 4], [2, 4], [2, 3]]);
    let p1 = vertices2Polygon([v1, v2, v3, v4]);
    let paths = p1.getPathBetween(v3, v4);
    let [vertices, edges] = paths.short;
    expect(vertices[0].equals(v3)).toBe(true);
    expect(vertices[1].equals(v4)).toBe(true);
    expect(edges[0].equals(p1.edges[2])).toBe(true);
    [vertices, edges] = paths.long;
    expect(vertices[0].equals(v3)).toBe(true);
    expect(vertices[1].equals(v2)).toBe(true);
    expect(vertices[2].equals(v1)).toBe(true);
    expect(vertices[3].equals(v4)).toBe(true);
    expect(edges[0].equals(p1.edges[1])).toBe(true);
    expect(edges[1].equals(p1.edges[0])).toBe(true);
    expect(edges[2].equals(p1.edges[3])).toBe(true);
  });
  it('Polygon getPathBetween (endIndex < startIndex)', function() {
    let [v1, v2, v3, v4] = list2Vertices([
      [0, 0], [0, 4], [2, 4], [2, 3]]);
    let p1 = vertices2Polygon([v1, v2, v3, v4]);
    let paths = p1.getPathBetween(v4, v3);
    let [vertices, edges] = paths.short;
    expect(vertices[0].equals(v4)).toBe(true);
    expect(vertices[1].equals(v3)).toBe(true);
    expect(edges[0].equals(p1.edges[2])).toBe(true);
    [vertices, edges] = paths.long;
    expect(vertices[0].equals(v4)).toBe(true);
    expect(vertices[1].equals(v1)).toBe(true);
    expect(vertices[2].equals(v2)).toBe(true);
    expect(vertices[3].equals(v3)).toBe(true);
    expect(edges[0].equals(p1.edges[3])).toBe(true);
    expect(edges[1].equals(p1.edges[0])).toBe(true);
    expect(edges[2].equals(p1.edges[1])).toBe(true);
  });
  it('Polygon pushPath case1', function() {
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
  it('Polygon pushPath case2', function() {
    let p1 = list2Polygon([[0, 0]]);
    let p2 = list2Polygon([[1, 2]]);
    let pTarget = list2Polygon([[0, 0], [1, 2]]);
    let vStart = new Vertex(0, 0);
    let vEnd = new Vertex(0, 0);
    p2.pushPath(p1, vStart, vEnd);
    expect(p2.equals(pTarget)).toBe(true);
  });
  it('Polygon pushPath case3', function() {
    let p1 = list2Polygon([
      [0, 0], [0, 4], [2, 4], [2, 3], [2, 2], [2, 1], [2, 0]]);
    p1.reverse();
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
  it('Polygon pushPath case4', function() {
    let p1 = list2Polygon([[1, 1], [2, 2], [3, 0]]);
    let p2 = new Polygon();
    let pTarget = list2Polygon([[1, 1], [2, 2]]);
    let vStart = new Vertex(1, 1);
    let vEnd = new Vertex(2, 2);
    p2.pushPath(p1, vStart, vEnd);
    expect(p2.equals(pTarget)).toBe(true);
  });
  it('Polygon pushPath case5', function() {
    let p1 = list2Polygon([[1, 1], [2, 2], [3, 0]]);
    let p2 = list2Polygon([[3, 0]]);
    let pTarget = list2Polygon([[1, 1], [2, 2], [3, 0]]);
    let vStart = new Vertex(1, 1);
    let vEnd = new Vertex(2, 2);
    p2.pushPath(p1, vStart, vEnd);
    expect(p2.equals(pTarget)).toBe(true);
  });
  it('Polygon pushPath case6', function() {
    let p1 = list2Polygon([
      [2, 0], [4, 0],
      [4, 4], [2, 4], [2, 3], [2, 2], [2, 1]]);
    let p2 = list2Polygon([[2, 4], [0, 4], [0, 0], [2, 0]]);
    let pTarget = list2Polygon([
      [2, 4], [0, 4],
      [0, 0], [2, 0], [2, 1], [2, 2], [2, 3]]);
    let vStart = new Vertex(2, 0);
    let vEnd = new Vertex(2, 4);
    p2.pushPath(p1, vStart, vEnd);
    expect(p2.equals(pTarget)).toBe(true);
  });
  it('Polygon serialization', function() {
    let p = randomPolygon();
    let p2 = p.copy();
    expect(p.equals(p2)).toBe(true);
    expect(p2 instanceof Polygon).toBe(true);
  });
});

describe('Shared Reference Behavior Tests', function() {
  it('Shared vertex movement', function() {
    let [v1, v2, v3, v4, v5, v6] = list2Vertices([
      [0, 0], [2, 0],
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
  it('Shared edges operations', function() {
    let [v1, v2, v3, v4, v5, v6] = list2Vertices([
      [0, 0], [2, 0],
      [4, 0], [4, 4], [2, 4], [0, 4]]);
    let newEdge = new Edge(v2, v5, EdgeTypes.BEZIER);
    newEdge.control_points[0].xy = [1, 1];
    newEdge.control_points[1].xy = [1, 3];
    let p1 = vertices2Polygon([v1, v2, v5, v6]);
    let p2 = vertices2Polygon([v2, v3, v4, v5]);
    let p3 = vertices2Polygon([v1, v2, v5, v6]);
    let p4 = vertices2Polygon([v2, v3, v4, v5]);
    p3.edges[1] = newEdge;
    p4.edges[3] = newEdge;
    // edge operation on p1
    p2.edges[3] = p1.edges[1]; // merge overlapping edge
    p1.edges[1].type = EdgeTypes.BEZIER;
    p1.edges[1].control_points[0].xy = [1, 1];
    p1.edges[1].control_points[1].xy = [1, 3];
    // check equality
    expect(p1.equals(p3)).toBe(true);
    expect(p2.equals(p4)).toBe(true);
  });
});
