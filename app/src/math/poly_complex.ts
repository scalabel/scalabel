import { Project } from "../types/project"
import Logger from "../server/logger"

/**
 * Checks all imported polygons and removes them if they have self-intersections
 *
 * @param project
 */
export default function filterPolygons(project: Project): [Project, string] {
  let msg: string = ""
  let count = 0
  const filteredLabels = []
  for (const [itemIndex, item] of project.items.entries()) {
    if (item.labels !== undefined) {
      for (const label of item.labels) {
        if (label.poly2d !== null) {
          let intersectionData = []
          for (const poly of label.poly2d) {
            intersectionData = polyIsComplex(poly.vertices)
            if (intersectionData.length > 0) {
              count += intersectionData.length
              for (const seg of intersectionData) {
                msg += `segment (${seg[0]}, ${seg[1]}, ${seg[2]}, ${
                  seg[3]
                }) intersects with segment (${seg[4]}, ${seg[5]}, ${seg[6]}, ${
                  seg[7]
                }) with polygon ID: ${label.id.toString()} in image with URL: ${
                  item.url !== undefined ? item.url.toString() : ""
                }\n`
              }
            } else {
              // Only keep valid polygons
              filteredLabels.push(label)
            }
          }
        }
      }
    }

    project.items[itemIndex].labels = filteredLabels
  }

  if (count > 0) {
    msg =
      `Found ${count} polygon intersection(s)!\n` +
      msg +
      "\nPlease check your import data."
    Logger.warning(msg)
  }

  return [project, msg]
}

/**
 * Determines if polygon has self-intersections and identifies all self-intersection coordinates
 *
 * @param vertices
 */
export function polyIsComplex(vertices: number[][]): number[][] {
  const intersections: number[][] = []

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
 * Determines if two lines intersect
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
  const det =
    (v2[0] - v1[0]) * (v4[1] - v3[1]) - (v4[0] - v3[0]) * (v2[1] - v1[1])
  if (det === 0) {
    return false
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
