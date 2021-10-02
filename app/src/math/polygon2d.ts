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
  // Closed polygon
  vertices.push(vertices[0])

  for (let i = 0; i < vertices.length - 1; i++) {
    for (let j = i + 1; j < vertices.length - 1; j++) {
      if (
        intersects(
          vertices[i],
          vertices[i + 1],
          vertices[j],
          vertices[j + 1]
        ) &&
        j - i !== 1 &&
        j - i !== vertices.length - 2
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
  // If the two vertices are the same, then they intersect
  if (v1 === v3 && v2 === v4) {
    return true
  }

  // Determines orthogonality of the line segments
  // If det = 0, line segments are parallel
  // If det = 1, line segments are orthogonal
  const det =
    (v2[0] - v1[0]) * (v4[1] - v3[1]) - (v4[0] - v3[0]) * (v2[1] - v1[1])

  if (det === 0) {
    // If line segments are colinear, then they intersect
    // Else they are parallel non-intersecting line segments
    return v2 === v3
  } else {
    const lambda =
      ((v4[1] - v3[1]) * (v4[0] - v1[0]) + (v3[0] - v4[0]) * (v4[1] - v1[1])) /
      det
    const gamma =
      ((v1[1] - v2[1]) * (v4[0] - v1[0]) + (v2[0] - v1[0]) * (v4[1] - v1[1])) /
      det
    return lambda >= 0 && lambda <= 1 && gamma >= 0 && gamma <= 1
  }
}
