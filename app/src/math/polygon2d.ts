/**
 * Merge nearby pixels if their distance is smaller than the threshold
 *
 * @param vertices
 * @param threshold
 */
export function mergeNearbyVertices(
  vertices: Array<[number, number]>,
  threshold: number
): Array<[number, number]> {
  const newVertices: Array<[number, number]> = []
  vertices.forEach((vertex) => {
    if (newVertices.length === 0) {
      newVertices.push(vertex)
    } else {
      const newVertex = newVertices[newVertices.length - 1]
      const dist =
        (vertex[0] - newVertex[0]) * (vertex[0] - newVertex[0]) +
        (vertex[1] - newVertex[1]) * (vertex[1] - newVertex[1])
      if (dist >= threshold * threshold) {
        newVertices.push(vertex)
      }
    }
  })
  return newVertices
}

/**
 * Determines if polygon has self-intersections and identifies all self-intersection coordinates
 *
 * @param vertices
 */
export function polyIsComplex(vertices: Array<[number, number]>): number[][] {
  const intersections: number[][] = []
  // Close polygon
  vertices.push(vertices[0])
  for (let i = 0; i < vertices.length - 1; i++) {
    for (let j = i + 1; j < vertices.length - 1; j++) {
      if (j - i === 1 || j - i === vertices.length - 2) {
        continue
      }
      if (
        intersects(vertices[i], vertices[i + 1], vertices[j], vertices[j + 1])
      ) {
        intersections.push([
          vertices[i][0],
          vertices[i][1],
          vertices[i + 1][0],
          vertices[i + 1][1],
          vertices[j][0],
          vertices[j][1],
          vertices[j + 1][0],
          vertices[j + 1][1]
        ])
      }
    }
  }
  return intersections
}

/**
 * Given three collinear points v1, v2, v3, the function checks if q lies
 * on line segment pr
 *
 * @param p
 * @param q
 * @param r
 */
function onSegment(p: number[], q: number[], r: number[]): boolean {
  if (
    q[0] <= Math.max(p[0], r[0]) &&
    q[0] >= Math.min(p[0], r[0]) &&
    q[1] <= Math.max(p[1], r[1]) &&
    q[1] >= Math.min(p[1], r[1])
  ) {
    return true
  }
  return false
}

/**
 * To find orientation of ordered triplet
 * The function returns following values
 * 0 -> p, q and r are collinear
 * 1 -> Clockwise
 * 2 -> Counterclockwise
 *
 * @param p
 * @param q
 * @param r
 */
function orientation(p: number[], q: number[], r: number[]): number {
  const val = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
  if (val === 0) {
    return 0
  } else if (val > 0) {
    return 1
  } else {
    return 2
  }
}

/**
 * Determines if two line segments intersect
 *
 * @param v1
 * @param v2
 * @param v3
 * @param v4
 */
function intersects(
  v1: number[],
  v2: number[],
  v3: number[],
  v4: number[]
): boolean {
  const o1 = orientation(v1, v2, v3)
  const o2 = orientation(v1, v2, v4)
  const o3 = orientation(v3, v4, v1)
  const o4 = orientation(v3, v4, v2)
  if (
    (o1 !== o2 && o3 !== o4) ||
    (o1 === 0 && onSegment(v1, v3, v2)) ||
    (o2 === 0 && onSegment(v1, v4, v2)) ||
    (o3 === 0 && onSegment(v3, v1, v4)) ||
    (o4 === 0 && onSegment(v3, v2, v4))
  ) {
    return true
  }
  return false
}
