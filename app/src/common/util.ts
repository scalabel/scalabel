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
export function estimateGroundPlane(vertices: number[]): THREE.Plane {
  const points: THREE.Vector3[] = []
  for (let i = 0; i < vertices.length; i += 3) {
    points.push(
      new THREE.Vector3(vertices[i], vertices[i + 1], vertices[i + 2])
    )
  }
  let bestPlane = new THREE.Plane()
  let maxNumPoints = 0
  const itMax = Math.ceil(points.length / 20)
  const threshold = 0.01
  for (let i = 0; i < itMax; i++) {
    const p1 = points[getRandomInt(points.length)]
    const p2 = points[getRandomInt(points.length)]
    const p3 = points[getRandomInt(points.length)]
    const plane = new THREE.Plane().setFromCoplanarPoints(p1, p2, p3)
    let numPoints = 0
    for (let p = 0; p < points.length; p++) {
      if (Math.abs(plane.distanceToPoint(points[p])) < threshold) {
        numPoints += 1
      }
    }
    if (numPoints > maxNumPoints) {
      bestPlane = plane
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

/**
 * Calculate rotation for estimated plane
 * Assume no rotation around z axis (up)
 *
 * @param baseNormal
 * @param estimatedNormal
 */
export function calculatePlaneRotation(
  baseNormal: THREE.Vector3,
  estimatedNormal: THREE.Vector3
): THREE.Vector3 {
  const rotation = new THREE.Quaternion().setFromUnitVectors(
    baseNormal,
    estimatedNormal
  )
  const rotationEuler = new THREE.Euler().setFromQuaternion(rotation)
  rotationEuler.z = 0
  return rotationEuler.toVector3()
}

/**
 * Calcuate estimated plane center.
 * Assume the plane center is directly below target center.
 *
 * @param plane
 * @param target
 */
export function calculatePlaneCenter(
  plane: THREE.Plane,
  target: THREE.Vector3
): THREE.Vector3 {
  const down = new THREE.Vector3(0, 0, -1)
  const ray = new THREE.Ray(target, down)
  const center = new THREE.Vector3()
  ray.intersectPlane(plane, center)
  return center
}
