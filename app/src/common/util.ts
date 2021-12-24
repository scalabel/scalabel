import { ADD_LABELS } from "../const/action"
import {
  ItemTypeName,
  LabelTypeName,
  ViewerConfigTypeName
} from "../const/common"
import { ActionPacketType } from "../types/message"
import * as THREE from "three"

/**
 * Handle invalid page request
 */
export function handleInvalidPage(): void {
  window.location.replace(window.location.origin)
}

/**
 * Get whether tracking is on
 * Also get the new item type
 *
 * @param itemType
 */
export function getTracking(itemType: string): [string, boolean] {
  switch (itemType) {
    case ItemTypeName.VIDEO:
      return [ItemTypeName.IMAGE, true]
    case ItemTypeName.POINT_CLOUD_TRACKING:
      return [ItemTypeName.POINT_CLOUD, true]
    case ItemTypeName.FUSION:
      return [ItemTypeName.FUSION, true]
    default:
      return [itemType, false]
  }
}

/**
 * Return viewer type required for given sensor and label types
 *
 * @param sensorType
 * @param labelTypes
 */
export function getViewerType(
  sensorType: ViewerConfigTypeName,
  labelTypes: LabelTypeName[]
): ViewerConfigTypeName {
  if (
    sensorType === ViewerConfigTypeName.IMAGE &&
    labelTypes.includes(LabelTypeName.BOX_3D)
  ) {
    return ViewerConfigTypeName.IMAGE_3D
  }
  return sensorType
}

/**
 * Create the link to the labeling instructions
 *
 * @param pageName
 */
function makeInstructionUrl(pageName: string): string {
  return `https://doc.scalabel.ai/instructions/${pageName}.html`
}

/**
 * Select the correct instruction url for the given label type
 *
 * @param labelType
 */
export function getInstructionUrl(labelType: string): string {
  switch (labelType) {
    case LabelTypeName.BOX_2D: {
      return makeInstructionUrl("bbox")
    }
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D: {
      return makeInstructionUrl("segmentation")
    }
    default: {
      return ""
    }
  }
}

/**
 * Select the correct page title for given label type
 *
 * @param labelType
 * @param itemType
 */
export function getPageTitle(labelType: string, itemType: string): string {
  const [, tracking] = getTracking(itemType)

  let title: string
  switch (labelType) {
    case LabelTypeName.TAG:
      title = "Image Tagging"
      break
    case LabelTypeName.BOX_2D:
      title = "2D Bounding Box"
      break
    case LabelTypeName.POLYGON_2D:
      title = "2D Segmentation"
      break
    case LabelTypeName.POLYLINE_2D:
      title = "2D Lane"
      break
    case LabelTypeName.BOX_3D:
      title = "3D Bounding Box"
      break
    default:
      title = ""
      break
  }
  if (tracking) {
    title = `${title} Tracking`
  }
  return title
}

/**
 * Converts index into a filename of size 6 with
 * trailing zeroes
 *
 * @param index
 */
export function index2str(index: number): string {
  return index.toString().padStart(6, "0")
}

/**
 * Checks if the action packet contains
 * any actions that would trigger a model query
 *
 * @param actionPacket
 * @param bots
 */
export function doesPacketTriggerModel(
  actionPacket: ActionPacketType,
  bots: boolean
): boolean {
  if (!bots) {
    return false
  }
  for (const action of actionPacket.actions) {
    if (action.type === ADD_LABELS) {
      return true
    }
  }
  return false
}

/** Ground plane estimation using RANSAC
 *
 * @param vertices
 */
export function estimateGroundPlane(vertices: number[]): number[] {
  const points: THREE.Vector3[] = []
  for (let i = 0; i < vertices.length; i += 3) {
    points.push(
      new THREE.Vector3(vertices[i], vertices[i + 1], vertices[i + 2])
    )
  }
  let bestPlane: number[] = []
  let maxNumPoints = 0
  const itMax = Math.ceil(points.length / 100)
  const threshold = 0.01
  for (let i = 0; i < itMax; i++) {
    const p1 = points[getRandomInt(points.length)]
    const p2 = points[getRandomInt(points.length)]
    const p3 = points[getRandomInt(points.length)]
    const plane = new THREE.Plane().setFromCoplanarPoints(p1, p2, p3)
    let numPoints = 0
    for (let p = 0; p < points.length; p += 5) {
      if (Math.abs(plane.distanceToPoint(points[p])) < threshold) {
        numPoints += 1
      }
    }
    if (numPoints > maxNumPoints) {
      bestPlane = [p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z]
      maxNumPoints = numPoints
    }
  }
  return bestPlane
}

/** Random integer generator
 *
 * @param max
 */
function getRandomInt(max: number): number {
  return Math.floor(Math.random() * max)
}
